from typing import Optional
import time
import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, degree

from PyNetAlign.data import Dataset
from .base_model import BaseModel


class REGAL(BaseModel):
    r"""REGAL algorithm for unsupervised pairwise attributed network alignment.
    Args:
        dataset (Dataset): PyNetAlign Dataset object containing the input graphs.
        gid1 (int): ID of the first graph for alignment.
        gid2 (int): ID of the second graph for alignment.
        use_attr (bool, optional): Whether to use node attributes for alignment. (default: :obj:`True`)
        dimension (int, optional): Dimension of the embedding space. (default: :obj:`128`)
        k (int, optional): Controls of landmarks to sample. (default: :obj:`10`)
        untillayer (int, optional): Calculation until the layer for xNetMF. (default: :obj:`2`)
        alpha (float, optional): Discount factor for further layers. (default: :obj:`0.01`)
        gammastruc (float, optional): Weight on structural similarity. (default: :obj:`1`)
        gammaattr (float, optional): Weight on attribute similarity. (default: :obj:`1`)
        numtop (int, optional): Number of top similarities to compute with kd-tree. If 0, computes all pairwise similarities. (default: :obj:`10`)
        buckets (int, optional): Base of log for degree (node feature) binning. (default: :obj:`2`)
        precision (int, optional): Precision of the computations. (default: :obj:`32`)
    """

    def __init__(self,
                 dataset: Dataset,
                 gid1: int,
                 gid2: int,
                 use_attr: Optional[bool] = True,
                 dimension: Optional[int] = 128,
                 k: Optional[int] = 10,
                 untillayer: Optional[int] = 2,
                 alpha: Optional[float] = 0.01,
                 gammastruc: Optional[float] = 1,
                 gammaattr: Optional[float] = 1,
                 numtop: Optional[int] = 10,
                 buckets: Optional[int] = 2,
                 precision: Optional[int] = 32):
        super(REGAL, self).__init__(precision=precision)

        assert isinstance(dataset, Dataset), 'Input dataset must be a PyNetAlign Dataset object'
        assert 0 <= gid1 < len(dataset.pyg_graphs) and 0 <= gid2 < len(dataset.pyg_graphs), 'Invalid graph IDs'
        assert gid1 != gid2, 'Cannot align a graph with itself'

        assert untillayer > 0, 'Number of layers must be greater than 0'
        assert buckets > 1, 'Number of buckets must be greater than 1'

        self.use_attr = use_attr
        self.dimension = dimension
        self.k = k
        self.untillayer = int(untillayer)
        self.alpha = alpha
        self.gammastruc = gammastruc
        self.gammaattr = gammaattr
        self.numtop = numtop
        self.buckets = int(buckets)

        self.graph1, self.graph2 = dataset.pyg_graphs[gid1], dataset.pyg_graphs[gid2]
        self.degree1 = degree(self.graph1.edge_index[0], num_nodes=self.graph1.num_nodes)
        self.degree2 = degree(self.graph2.edge_index[0], num_nodes=self.graph2.num_nodes)
        self.max_degree = max(self.degree1.max().item(), self.degree2.max().item())

    def forward(self, *args, **kwargs):
        return self.get_xnetmf_embedding()

    def get_xnetmf_embedding(self):
        print('Generating structural embeddings...', end=' ')
        struct_emb1 = self.get_structural_embedding(self.graph1, self.max_degree)
        struct_emb2 = self.get_structural_embedding(self.graph2, self.max_degree)
        print('Done')

        print('Generating xNetMF embeddings...', end=' ')
        n1, n2 = self.graph1.num_nodes, self.graph2.num_nodes
        num_landmarks = min(int(self.k * np.log(n1 + n2) / np.log(2)), n1 + n2)
        sampled_landmarks = torch.randperm(n1 + n2)[:num_landmarks]

        # Merge embeddings of the two graphs for effective similarity computation
        struct_emb = torch.vstack([struct_emb1, struct_emb2])
        landmarks_struct_emb = struct_emb[sampled_landmarks]
        struct_dist = torch.norm(struct_emb[:, None, :] - landmarks_struct_emb[None, :, :], dim=2)
        if self.use_attr and self.graph1.x is not None and self.graph2.x is not None:
            attribute_emb = torch.vstack([self.graph1.x, self.graph2.x])
            landmarks_attr_emb = attribute_emb[sampled_landmarks]
            attribute_dist = (attribute_emb[:, None, :] != landmarks_attr_emb[None, :, :]).to(self.precision).sum(dim=2)
        else:
            attribute_dist = 0
        # TODO: Test sensitivity for different choices of "distance" computation
        C = torch.exp(-(self.gammastruc * struct_dist + self.gammaattr * attribute_dist))

        W_pinv = torch.pinverse(C[sampled_landmarks])
        U, X, V = torch.svd(W_pinv)
        Wfac = U @ torch.diag(torch.sqrt(X))
        xnetmf_emb = C @ Wfac
        xnetmf_emb = F.normalize(xnetmf_emb, p=2, dim=1)
        print('Done')

        return xnetmf_emb[:n1], xnetmf_emb[n1:]

    def get_structural_embedding(self, graph, max_degree):
        degrees = degree(graph.edge_index[0], num_nodes=graph.num_nodes)
        assert degrees.max().item() <= max_degree, 'Max degree is less than the maximum degree in the graph'

        neighborhood = self.get_k_hop_neighbors(graph, self.untillayer)
        num_features = int(np.log(max_degree) / np.log(self.buckets)) + 1

        # TODO: Optimize this part (1. vectorized manner may be possible, 2. get_k_hop_neighbors may not be necessary)
        structural_embedding = torch.zeros(graph.num_nodes, num_features)
        for n in range(graph.num_nodes):
            for layer in neighborhood[n].keys():
                neighbors = neighborhood[n][layer]
                if len(neighbors) > 0:
                    # get degree sequence
                    neighbors_degrees = degrees[neighbors]
                    filtered_neighbors_degrees = neighbors_degrees[neighbors_degrees > 0]
                    index_vec = (torch.log(filtered_neighbors_degrees) / np.log(self.buckets)).to(torch.int)
                    structural_embedding[n, :] += torch.bincount(index_vec, minlength=num_features) * (self.alpha ** layer)

        return structural_embedding

    @staticmethod
    def get_k_hop_neighbors(graph, k):
        adj = to_dense_adj(graph.edge_index).squeeze().to(torch.bool)

        neighborhoods = {node: {0: torch.tensor([node])} for node in range(graph.num_nodes)}
        for node in range(graph.num_nodes):
            last_neighbor_vec = torch.zeros(graph.num_nodes, dtype=torch.bool)
            last_neighbor_vec[node] = True
            for layer in range(1, k + 1):
                neighbor_vec = (adj[neighborhoods[node][layer - 1]].sum(dim=0) > 0) & ~last_neighbor_vec
                if neighbor_vec.sum() == 0:
                    break
                neighborhoods[node][layer] = torch.where(neighbor_vec)[0]
                last_neighbor_vec |= neighbor_vec

        return neighborhoods
