from typing import Optional
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj

from PyNetAlign.data import Dataset
from PyNetAlign.datasets import PhoneEmail
from PyNetAlign.utils import get_anchor_pairs
from .base_model import BaseModel


class IsoRank(BaseModel):
    r"""IsoRank algorithm for pairwise plain network alignment."""
    def __init__(self,
                 dataset: Dataset,
                 gid1: int,
                 gid2: int,
                 alpha: Optional[float] = 0.4,
                 precision: Optional[int] = 32):
        super(IsoRank, self).__init__(precision=precision)

        assert isinstance(dataset, Dataset), 'Input dataset must be a PyNetAlign Dataset object'
        assert 0 <= gid1 < len(dataset.pyg_graphs) and 0 <= gid2 < len(dataset.pyg_graphs), 'Invalid graph IDs'
        assert gid1 != gid2, 'Cannot align a graph with itself'
        assert 0 <= alpha <= 1, 'Alpha must be in [0, 1]'

        self.n1 = dataset.pyg_graphs[gid1].num_nodes
        self.n2 = dataset.pyg_graphs[gid2].num_nodes
        self.alpha = alpha

        adj1 = to_dense_adj(dataset.pyg_graphs[gid1].edge_index).squeeze()
        adj2 = to_dense_adj(dataset.pyg_graphs[gid2].edge_index).squeeze()
        self.adj1 = F.normalize(adj1, p=1, dim=0)
        self.adj2 = F.normalize(adj2, p=1, dim=0)

        self.H = torch.zeros(self.n1, self.n2, dtype=self.precision)
        self.anchors = get_anchor_pairs(dataset.train_data, gid1, gid2)
        self.H[self.anchors[:, 0], self.anchors[:, 1]] = 1

        self.S = torch.rand(self.n1, self.n2, dtype=self.precision)

    def forward(self):
        self.S = self.alpha * self.adj1 @ self.S @ self.adj2.T + (1 - self.alpha) * self.H
        return self.S

    def to(self, device):
        self.H = self.H.to(device)
        self.S = self.S.to(device)
        return self
