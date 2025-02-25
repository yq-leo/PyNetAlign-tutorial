from typing import Optional, List, Tuple, Union
from collections import defaultdict
import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import degree

from PyNetAlign.data import Dataset
from .base_model import BaseModel


class CrossMNA(BaseModel):
    r"""CrossMNA algorithm for plain multi-network alignment.
    Args:
        dataset (Dataset): PyNetAlign Dataset object containing the input graphs.
        gids (list): List of graph IDs for alignment.
    """

    def __init__(self,
                 dataset: Dataset,
                 gids: Union[List[int], Tuple[int, ...]],
                 batch_size: Optional[int] = 512 * 8,
                 neg_samples: Optional[int] = 1,
                 node_emb_dims: Optional[int] = 200,
                 graph_emb_dims: Optional[int] = 100,
                 lr: Optional[float] = 0.02,
                 precision: Optional[int] = 32):
        super(CrossMNA, self).__init__(precision=precision)

        assert isinstance(dataset, Dataset), 'Input dataset must be a PyNetAlign Dataset object'
        assert len(gids) > 1, 'At least two graphs are required for alignment'
        for gid in gids:
            assert 0 <= gid < len(dataset.pyg_graphs), 'Invalid graph ID'
        assert len(set(gids)) == len(gids), 'Graph IDs must be unique'

        self.gids = gids
        self.graphs = [dataset.pyg_graphs[gid] for gid in gids]
        self.anchor_links = dataset.train_data[:, gids]

        self.batch_size = batch_size
        self.neg_samples = neg_samples
        self.node_emb_dims = node_emb_dims
        self.graph_emb_dims = graph_emb_dims
        self.lr = lr

        # Initialization
        self.alias_samplers_list = []
        for gid in gids:
            graph = dataset.pyg_graphs[gid]
            alias_sampler = self.init_sampler(graph)
            self.alias_samplers_list.append(alias_sampler)

        self.id2node, self.node2id = self.merge_graphs_on_anchors()
        self.model, self.optimizer = self.init_model(len(self.id2node))

    def forward(self):
        train_samples = self.generate_samples()

        total_loss = 0
        for u_i, u_j, label, gid_vec in train_samples:
            u_i = u_i.cpu().numpy()
            u_j = u_j.cpu().numpy()
            gid_vec = gid_vec.cpu().numpy()

            mapped_u_i = np.array([self.node2id[(gid, u)] for gid, u in zip(gid_vec, u_i)])
            mapped_u_j = np.array([self.node2id[(gid, u)] for gid, u in zip(gid_vec, u_j)])

            self.optimizer.zero_grad()
            loss = self.model(mapped_u_i, mapped_u_j, gid_vec, label)
            loss.backward()
            self.optimizer.step()
            total_loss += loss

        with torch.no_grad():
            out_embs_dict = {}
            embeddings = F.normalize(self.model.embedding, p=2, dim=1)
            for gid, graph in enumerate(self.graphs):
                embs = embeddings[[self.node2id[(gid, u)] for u in range(graph.num_nodes)]]
                out_embs_dict[self.gids[gid]] = embs

        return out_embs_dict, total_loss

    def generate_samples(self):
        all_edges = torch.empty((2, 0), dtype=torch.int64)
        all_gid_vec = torch.empty((0,), dtype=torch.int)
        all_neg_samples = torch.empty((self.neg_samples, 0), dtype=torch.int64)
        for gid, graph in enumerate(self.graphs):
            all_gid_vec = torch.cat([all_gid_vec, torch.tensor([gid] * graph.num_edges, dtype=torch.int)])
            all_edges = torch.hstack([all_edges, graph.edge_index])
            neg_samples = self.alias_samplers_list[gid].sample(num_samples=self.neg_samples * graph.num_edges).reshape(self.neg_samples, graph.num_edges)
            all_neg_samples = torch.hstack([all_neg_samples, torch.from_numpy(neg_samples)])

        # Shuffle sampled node pairs
        num_all_edges = all_edges.shape[1]
        perm = torch.randperm(num_all_edges)
        all_edges = all_edges[:, perm]
        all_gid_vec = all_gid_vec[perm]
        all_neg_samples = all_neg_samples[:, perm]

        ui_samples = torch.repeat_interleave(all_edges[0, :], repeats=self.neg_samples+1, dim=0)
        uj_samples = torch.vstack([all_edges[1, :], all_neg_samples]).T.flatten()
        gid_samples = torch.repeat_interleave(all_gid_vec, repeats=self.neg_samples+1)
        label_samples = torch.vstack([torch.ones(num_all_edges), -torch.ones(self.neg_samples, num_all_edges)]).T.flatten()

        assert ui_samples.shape == uj_samples.shape == gid_samples.shape == label_samples.shape, 'Shape mismatch'

        # Divide sampled node pairs into batches
        batched_samples = []
        num_batches = ui_samples.shape[0] // self.batch_size
        for i in range(num_batches):
            left = i * self.batch_size
            right = (i + 1) * self.batch_size
            u_i = ui_samples[left:right]
            u_j = uj_samples[left:right]
            label = label_samples[left:right]
            gid_vec = gid_samples[left:right]
            batched_samples.append((u_i, u_j, label, gid_vec))

        return batched_samples

    def init_model(self, num_nodes_in_merged_graph):
        model = MultiNetworkEmb(num_of_nodes=num_nodes_in_merged_graph,
                                num_layer=len(self.graphs),
                                batch_size=self.batch_size,
                                K=self.neg_samples,
                                node_emb_dims=self.node_emb_dims,
                                layer_emb_dims=self.graph_emb_dims)
        optimizer = torch.optim.RMSprop(model.parameters(), lr=self.lr, alpha=0.99, eps=1.0, centered=True, momentum=0.0)
        return model, optimizer

    @staticmethod
    def init_sampler(graph):
        node_positive_distribution = degree(graph.edge_index[0], num_nodes=graph.num_nodes) ** 0.75
        node_positive_distribution /= node_positive_distribution.sum()
        return AliasSampler(prob=node_positive_distribution)

    def merge_graphs_on_anchors(self):
        anchor_maps = dict()
        anchor_links = self.anchor_links.cpu().numpy()
        for anchor_link in anchor_links:
            true_anchor = None
            for gid, anchor in enumerate(anchor_link):
                if anchor > -1:
                    if true_anchor is None:
                        true_anchor = (gid, int(anchor))
                    else:
                        anchor_maps[(gid, int(anchor))] = true_anchor

        id2node, node2id = defaultdict(list), dict()
        for gid, g in enumerate(self.graphs):
            for node in range(g.num_nodes):
                if (gid, node) in anchor_maps:
                    nid = node2id[anchor_maps[(gid, node)]]
                else:
                    nid = len(id2node)
                node2id[(gid, node)] = nid
                id2node[nid].append((gid, node))

        return id2node, node2id


class MultiNetworkEmb(nn.Module):
    def __init__(self, num_of_nodes, batch_size, K, node_emb_dims, num_layer, layer_emb_dims):
        super(MultiNetworkEmb, self).__init__()
        self.batch_size = batch_size
        self.K = K

        # Parameters
        self.embedding = nn.Parameter(torch.empty(num_of_nodes, node_emb_dims))
        self.L_embedding = nn.Parameter(torch.empty(num_layer + 1, layer_emb_dims))
        self.W = nn.Parameter(torch.empty(node_emb_dims, layer_emb_dims))

        # Initialize with truncated normal (approximation using normal distribution)
        nn.init.trunc_normal_(self.embedding, mean=0.0, std=0.3)
        nn.init.trunc_normal_(self.L_embedding, mean=0.0, std=0.3)
        nn.init.trunc_normal_(self.W, mean=0.0, std=0.3)

        # Normalize embeddings
        self.embedding.data = F.normalize(self.embedding.data, p=2, dim=1)
        self.L_embedding.data = F.normalize(self.L_embedding.data, p=2, dim=1)
        self.W.data = F.normalize(self.W.data, p=2, dim=1)

    def forward(self, u_i, u_j, this_layer, label):
        # Step 1: Look up embeddings
        u_i_embedding = self.embedding[u_i]
        u_j_embedding = self.embedding[u_j]

        # Step 2: W * u
        u_i_embedding = torch.matmul(u_i_embedding, self.W)
        u_j_embedding = torch.matmul(u_j_embedding, self.W)

        # Step 3: Look up layer embedding
        l_i_embedding = self.L_embedding[this_layer]

        # Step 4: r_i = u_i * W + l
        r_i = u_i_embedding + l_i_embedding
        r_j = u_j_embedding + l_i_embedding

        # Step 5: Compute inner product
        inner_product = torch.sum(r_i * r_j, dim=1)

        # Loss function
        loss = -torch.sum(F.logsigmoid(label * inner_product))

        return loss


class AliasSampler:
    def __init__(self, prob):
        self.n = len(prob)
        self.U = np.array(prob) * self.n
        self.K = [i for i in range(len(prob))]
        overfull, underfull = [], []
        for i, U_i in enumerate(self.U.tolist()):
            if U_i > 1:
                overfull.append(i)
            elif U_i < 1:
                underfull.append(i)
        while len(overfull) and len(underfull):
            i, j = overfull.pop(), underfull.pop()
            self.K[j] = i
            self.U[i] = self.U[i] - (1 - self.U[j])
            if self.U[i] > 1:
                overfull.append(i)
            elif self.U[i] < 1:
                underfull.append(i)

    def sample(self, num_samples=1):
        x = np.random.rand(num_samples)
        i = np.floor(self.n * x).astype(np.int32)
        y = self.n * x - i
        samples = np.array([i[k] if y[k] < self.U[i[k]] else self.K[i[k]] for k in range(num_samples)])
        return samples


