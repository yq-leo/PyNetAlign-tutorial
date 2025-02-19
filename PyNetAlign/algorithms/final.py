from typing import Optional
import time
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj

from PyNetAlign.data import Dataset
from PyNetAlign.utils import get_anchor_pairs
from .base_model import BaseModel


class FINAL(BaseModel):
    r"""FINAL algorithm for pairwise attributed network alignment."""

    def __init__(self,
                 dataset: Dataset,
                 gid1: int,
                 gid2: int,
                 alpha: Optional[float] = 0.5,
                 precision: Optional[int] = 32):
        super(FINAL, self).__init__(precision=precision)

        assert isinstance(dataset, Dataset), 'Input dataset must be a PyNetAlign Dataset object'
        assert gid1 < len(dataset.pyg_graphs) and gid2 < len(dataset.pyg_graphs), 'Invalid graph IDs'
        assert 0 <= alpha <= 1, 'Alpha must be in [0, 1]'

        self.alpha = alpha

        graph1, graph2 = dataset.pyg_graphs[gid1], dataset.pyg_graphs[gid2]
        self.n1, self.n2 = graph1.num_nodes, graph2.num_nodes
        self.adj1 = to_dense_adj(graph1.edge_index).squeeze().to(self.precision)
        self.adj2 = to_dense_adj(graph2.edge_index).squeeze().to(self.precision)
        self.node_attr1, self.node_attr2 = self.init_node_feat(graph1), self.init_node_feat(graph2)
        self.edge_attr1_adj, self.edge_attr2_adj = self.init_edge_feat_adj(graph1), self.init_edge_feat_adj(graph2)
        self.anchors = get_anchor_pairs(dataset.train_data, gid1, gid2)

        self.N = torch.zeros(self.n2, self.n1, dtype=self.precision)
        self.d = torch.zeros(self.n2, self.n1, dtype=self.precision)
        # TODO: compute h by degree similarity
        self.h = torch.zeros(self.n2, self.n1, dtype=self.precision)
        self.h[self.anchors[:, 1], self.anchors[:, 0]] = 1
        self.s = torch.clone(self.h)

    def preprocess(self):
        num_node_attr = self.node_attr1.shape[1]
        num_edge_attr = self.edge_attr1_adj.shape[0]

        # Compute node feature cosine cross-similarity
        for k in range(num_node_attr):
            self.N += torch.outer(self.node_attr2[:, k], self.node_attr1[:, k])

        # Compute the Kronecker degree vector
        start_time = time.time()
        for i in range(num_edge_attr):
            for j in range(num_node_attr):
                self.d += torch.outer((self.edge_attr2_adj[i] * self.adj2 @ self.node_attr2[:, j]),
                                      (self.edge_attr1_adj[i] * self.adj1 @ self.node_attr1[:, j]))
        print('Time for degree: {:.2f} seconds'.format(time.time() - start_time))

        D = self.N * self.d
        maskD = D > 0
        D[maskD] = torch.reciprocal(torch.sqrt(D[maskD]))

        self.N = self.N * D

    def forward(self, *args, **kwargs):
        prev_s = torch.clone(self.s)
        M = self.N * self.s
        S = torch.zeros_like(self.N)

        for i in range(self.node_attr1.shape[1]):
            S += (self.edge_attr2_adj[i] * self.adj2) @ M @ (self.edge_attr1_adj[i] * self.adj1)

        self.s = (1 - self.alpha) * self.h + self.alpha * self.N * S
        diff = torch.norm(self.s - prev_s)
        return self.s, diff

    def init_node_feat(self, graph):
        if graph.num_node_features == 0:
            node_attr = torch.ones(graph.num_nodes, 1, dtype=self.precision).to(graph.device)
        else:
            node_attr = graph.x.to(self.precision)
        return F.normalize(node_attr, p=2, dim=1)

    def init_edge_feat_adj(self, graph):
        if graph.edge_attr is None:
            edge_attr = torch.ones(graph.edge_index.shape[1], 1, dtype=self.precision).to(graph.device)
        else:
            edge_attr = graph.edge_attr.to(self.precision)
        edge_attr = F.normalize(edge_attr, p=2, dim=1)

        edge_attr_adj = torch.zeros(graph.edge_index.shape[1], graph.num_nodes, graph.num_nodes, dtype=self.precision)
        edge_attr_adj[:, graph.edge_index[0], graph.edge_index[1]] = edge_attr.T

        return edge_attr_adj

    def to(self, device):
        self.N = self.N.to(device)
        self.d = self.d.to(device)
        self.h = self.h.to(device)
        self.s = self.s.to(device)
        return self
