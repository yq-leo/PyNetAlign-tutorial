from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.utils import degree

from PyNetAlign.data import Dataset
from PyNetAlign.utils import get_anchor_pairs
from PyNetAlign.algorithms.base_model import BaseModel
from .model import NetTransModel
from .data import ContextDataset
from .sampling import *


class NetTrans(BaseModel):
    r"""NetTrans algorithm for pairwise network alignment via network transformation."""

    def __init__(self,
                 dataset: Dataset,
                 gid1: int,
                 gid2: int,
                 hid_dim: Optional[int] = 128,
                 depth: Optional[int] = 2,
                 pooling_ratio: Optional[float] = 0.2,
                 attr_coeff: Optional[float] = 1.,
                 adj_coeff: Optional[float] = 1.,
                 rank_coeff: Optional[float] = 1.,
                 margin: Optional[float] = 1.,
                 neg_size: Optional[int] = 20,
                 batch_size: Optional[int] = 300,
                 lr: Optional[float] = 0.001,
                 temperature: Optional[float] = 1.,
                 min_temperture: Optional[float] = 0.1,
                 anneal_rate: Optional[float] = 2e-5,
                 seed: Optional[int] = 123,
                 precision: Optional[int] = 32):
        super(NetTrans, self).__init__(precision=precision)

        assert isinstance(dataset, Dataset), 'Input dataset must be a PyNetAlign Dataset object'
        assert 0 <= gid1 < len(dataset.pyg_graphs) and 0 <= gid2 < len(dataset.pyg_graphs), 'Invalid graph IDs'
        assert gid1 != gid2, 'Cannot align a graph with itself'

        self.graph1, self.graph2 = dataset.pyg_graphs[gid1], dataset.pyg_graphs[gid2]
        self.n1, self.n2 = self.graph1.num_nodes, self.graph2.num_nodes
        self.anchor_links = get_anchor_pairs(dataset.train_data, gid1, gid2)

        self.hid_dim = hid_dim
        self.depth = depth
        self.pooling_ratio = pooling_ratio
        self.attr_coeff = attr_coeff
        self.adj_coeff = adj_coeff
        self.rank_coeff = rank_coeff
        self.margin = margin
        self.neg_size = neg_size
        self.batch_size = batch_size
        self.temperature = temperature
        self.min_temperature = min_temperture
        self.anneal_rate = anneal_rate

        # Initialization
        anchor_embeddings1, anchor_embeddings2 = self.get_anchor_embeddings()
        if self.graph1.x is not None and self.graph2.x is not None:
            self.node_attr1 = torch.cat([self.graph1.x, anchor_embeddings1], dim=1)
            self.node_attr2 = torch.cat([self.graph2.x, anchor_embeddings2], dim=1)
        else:
            self.node_attr1 = anchor_embeddings1
            self.node_attr2 = anchor_embeddings2

        self.edge_weight1 = torch.ones(self.graph1.edge_index.shape[1], dtype=self.precision, device=self.device)
        self.edge_weight2 = torch.ones(self.graph2.edge_index.shape[1], dtype=self.precision, device=self.device)

        anchor_context_pairs1 = self.sample_anchor_neighbor_pairs(self.graph1, self.anchor_links[:, 0])
        anchor_context_pairs2 = self.sample_anchor_neighbor_pairs(self.graph2, self.anchor_links[:, 1])
        self.anchor_context_pairs1, self.anchor_context_pairs2 = self.balance_samples(anchor_context_pairs1,
                                                                                      anchor_context_pairs2)
        self.neg_context_prob1, self.anchor_map1 = self.get_neg_context_prob(self.graph1, self.anchor_links[:, 0])
        self.neg_context_prob2, self.anchor_map2 = self.get_neg_context_prob(self.graph2, self.anchor_links[:, 1])

        # Model
        anchor_context_dataset = ContextDataset(self.anchor_context_pairs1, self.anchor_context_pairs2)
        self.data_loader = DataLoader(dataset=anchor_context_dataset, batch_size=self.batch_size, shuffle=True)

        torch.manual_seed(seed)
        # if torch.cuda.is_available():
        #     torch.cuda.manual_seed(seed)

        self.model = NetTransModel(in_dim=self.node_attr1.shape[1],
                                   hid_dim=self.hid_dim,
                                   out_dim=self.node_attr2.shape[1],
                                   pooling_ratio=self.pooling_ratio,
                                   depth=self.depth,
                                   margin=self.margin).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.epoch_cnt = 0

    def forward(self):
        self.model.train()
        if self.epoch_cnt % 10 == 1:
            self.temperature = max(self.temperature * np.exp(-self.anneal_rate * self.epoch_cnt),
                                   self.min_temperature)
        for i, (context1, context2) in enumerate(self.data_loader):
            context1, context2 = context1.to(self.device), context2.to(self.device)
            self.optimizer.zero_grad()
            for name, param in self.model.named_parameters():
                if param.isnan().any():
                    print(name)
            x, sup1, y, sup2, assign_mats, recon_y = self.model(x=self.node_attr1,
                                                                edge_index=self.graph1.edge_index,
                                                                edge_weight=self.edge_weight1,
                                                                y=self.node_attr2,
                                                                edge_index_y=self.graph2.edge_index,
                                                                edge_weight_y=self.edge_weight2,
                                                                anchor_links=self.anchor_links.T,
                                                                temperature=self.temperature)
            # negative sampling
            with torch.no_grad():
                anchor_nodes1 = context1[:, 0].reshape(-1)
                pos_context_nodes1 = context1[:, 1].reshape(-1)
                anchor_nodes2 = context2[:, 0].reshape(-1)
                pos_context_nodes2 = context2[:, 1].reshape(-1)

                negs1, negs2 = uniform_negative_sampling(anchor_nodes1, anchor_nodes2, self.n1, self.n2, self.neg_size)
                neg_context1 = negative_edge_sampling(self.neg_context_prob1, self.anchor_map1[anchor_nodes1], self.neg_size)
                neg_context2 = negative_edge_sampling(self.neg_context_prob2, self.anchor_map2[anchor_nodes2], self.neg_size)

            negs1, negs2 = negs1.flatten().to(self.device), negs2.flatten().to(self.device)
            neg_context1, neg_context2 = neg_context1.flatten().to(self.device), neg_context2.flatten().to(self.device)

            neg_emb1, neg_emb2 = y[negs1], x[negs2]
            neg_context_emb1, neg_context_emb2 = x[neg_context1], y[neg_context2]
            anchor_emb1, anchor_emb2 = x[anchor_nodes1], y[anchor_nodes2]
            pos_emb1, pos_emb2 = x[pos_context_nodes1], y[pos_context_nodes2]

            adj_loss = self.model.adj_loss(anchor_emb1, pos_emb1, neg_context_emb1) + self.model.adj_loss(anchor_emb2, pos_emb2, neg_context_emb2)
            align_loss = self.model.align_loss(anchor_emb1, anchor_emb2, neg_emb1, neg_emb2)
            total_loss = self.adj_coeff * adj_loss + self.rank_coeff * align_loss

            print(f"Epoch:{self.epoch_cnt + 1}, Batch:{i + 1}/{len(self.data_loader)}, Training loss:{round(total_loss.item(), 4)}")

            total_loss.backward()
            self.optimizer.step()

        self.epoch_cnt += 1
        self.model.eval()
        emb1 = F.normalize(x.detach(), p=2, dim=1)
        emb2 = F.normalize(y.detach(), p=2, dim=1)
        return emb1, emb2

    def get_anchor_embeddings(self):
        num_anchors = self.anchor_links.shape[0]
        anchor_embeddings1 = torch.zeros(self.graph1.num_nodes, num_anchors, dtype=self.precision)
        anchor_embeddings2 = torch.zeros(self.graph2.num_nodes, num_anchors, dtype=self.precision)
        anchor_embeddings1[self.anchor_links[:, 0], torch.arange(num_anchors)] = 1
        anchor_embeddings2[self.anchor_links[:, 1], torch.arange(num_anchors)] = 1
        return anchor_embeddings1, anchor_embeddings2

    @staticmethod
    def sample_anchor_neighbor_pairs(graph, anchors):
        degrees = degree(graph.edge_index[0], num_nodes=graph.num_nodes)

        sampled_context_pairs = torch.empty((0, 2), dtype=torch.int64)
        for node in anchors:
            neighbors = graph.edge_index[1, graph.edge_index[0] == node]
            if len(neighbors) > 100:
                p = degrees[neighbors] / degrees[neighbors].sum()
                neighbors = neighbors[torch.multinomial(p, 100, replacement=True)]
            context = torch.vstack([torch.tensor([node] * len(neighbors)), neighbors]).T
            sampled_context_pairs = torch.vstack([sampled_context_pairs, context])

        # Shuffle the context pairs
        sampled_context_pairs = sampled_context_pairs[torch.randperm(sampled_context_pairs.size(0))]
        return sampled_context_pairs

    @staticmethod
    def balance_samples(sample_pairs1, sample_pairs2):
        if len(sample_pairs1) < len(sample_pairs2):
            len_diff = len(sample_pairs2) - len(sample_pairs1)
            idx = torch.randint(len(sample_pairs1), (len_diff,))
            imputes = sample_pairs1[idx]
            balanced_sample_pairs1 = torch.vstack([sample_pairs1, imputes])
            balanced_sample_pairs2 = sample_pairs2
        else:
            len_diff = len(sample_pairs1) - len(sample_pairs2)
            idx = torch.randint(len(sample_pairs2), (len_diff,))
            imputes = sample_pairs2[idx]
            balanced_sample_pairs2 = torch.vstack([sample_pairs2, imputes])
            balanced_sample_pairs1 = sample_pairs1
        return balanced_sample_pairs1, balanced_sample_pairs2

    def get_neg_context_prob(self, graph, anchors):
        prob = torch.ones((len(anchors), graph.num_nodes), dtype=self.precision)
        for i, anchor in enumerate(anchors):
            neighbors = graph.edge_index[1, graph.edge_index[0] == anchor]
            prob[i][neighbors] = 0

        anchor_node_map = -1 * torch.ones(graph.num_nodes, dtype=torch.int64)
        anchor_node_map[anchors] = torch.arange(len(anchors), dtype=torch.int64)

        return prob, anchor_node_map
