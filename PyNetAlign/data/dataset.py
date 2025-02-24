import os
from pathlib import Path
from typing import Optional, Union
import torch
from torch_geometric.data import Data


class Dataset:
    r"""A dataset object storing multiple graphs and ground-truth anchor links for alignment.
    """
    def __init__(self,
                 root: Union[str, Path],
                 name: str,
                 train_ratio: Optional[float] = 0.2,
                 precision: Optional[int] = 32,
                 seed: Optional[int] = 0):
        assert 0 < train_ratio < 1, 'Training ratio must be in (0, 1)'
        assert precision in [32, 64], 'Precision must be either 32 or 64'

        self.root = Path(root)
        self.name = name
        self.train_ratio = train_ratio
        self.seed = seed
        self.precision = torch.float32 if precision == 32 else torch.float64

        self.pyg_graphs, self._anchor_links = self.load_dataset()
        self._validate()
        self.train_data, self.test_data = self.train_test_split()

    def load_dataset(self):
        data_dict = torch.load(Path.joinpath(self.root, f'{self.name}.pt'), weights_only=True)
        pyg_graphs = list()
        for gid, gname in enumerate(data_dict['graphs']):
            num_nodes = data_dict['number_of_nodes'][gid]
            x = data_dict['node_attributes'][gid].to(self.precision) if 'node_attributes' in data_dict else None
            edge_attr = data_dict['edge_attributes'][gid].to(self.precision) if 'edge_attributes' in data_dict else None
            edge_index = data_dict['edges'][gid]
            pyg_graph = Data(name=gname, num_nodes=num_nodes, x=x, edge_index=edge_index, edge_attr=edge_attr)
            pyg_graphs.append(pyg_graph)
        anchor_links = data_dict['anchor_links']
        return pyg_graphs, anchor_links

    def train_test_split(self):
        num_anchor = self._anchor_links.shape[0]
        rng_state = torch.get_rng_state()
        torch.manual_seed(self.seed)
        perm = torch.randperm(num_anchor)
        torch.set_rng_state(rng_state)
        train_size = int(num_anchor * self.train_ratio)
        return self._anchor_links[perm[:train_size]], self._anchor_links[perm[train_size:]]

    def _check_integrity(self, root):
        raise NotImplementedError

    def _validate(self):
        assert hasattr(self, 'pyg_graphs') and hasattr(self, '_anchor_links'), 'Dataset has not been loaded yet, wrong place for validation'
        assert type(self.pyg_graphs) in [list, tuple], 'Graphs must be stored in a list or tuple'
        assert len(self.pyg_graphs) > 1, 'At least two graphs are required for alignment'
        assert all([isinstance(g, Data) for g in self.pyg_graphs]), 'Each graph must be a PyG Data object'
        assert isinstance(self._anchor_links, torch.Tensor), 'Anchor links must be stored in a PyTorch tensor'
        assert self._anchor_links.dim() == 2, 'Anchor links must be a 2D tensor'
        assert len(self.pyg_graphs) == self._anchor_links.shape[1], 'Number of PyG graphs and anchor links dimension do not match'
        for i, g in enumerate(self.pyg_graphs):
            assert g.edge_index.max() < g.num_nodes, f'Node index must be less than the number of nodes in graph {i}'
            assert self._anchor_links[:, i].max() < g.num_nodes, f'Anchor link must be less than the number of nodes in graph {i}'
            if g.x is not None:
                assert g.x.shape[0] == g.num_nodes, f'Number of nodes and node attributes must match in graph {i}'
            if g.edge_attr is not None:
                assert g.edge_attr.shape[0] == g.edge_index.shape[1], f'Number of edges and edge attributes must match in graph {i}'

    def __str__(self):
        network_count = len(self.pyg_graphs)  # Number of networks (Assuming stored in a list)
        # Header
        output = (
                f"Dataset: {self.name}\n"
                f"{'=' * 60}\n"
                f"{'Graphs':<20}" + "".join([f"{self.pyg_graphs[i].name:>15}" for i in range(network_count)]) +
                f"\n{'-' * 60}\n"
        )
        # Number of Nodes per graph
        output += (
                f"{'# Nodes':<20}" +
                "".join([f"{self.pyg_graphs[i].num_nodes:>15,}" for i in range(network_count)]) + "\n"
        )
        # Number of Edges per graph
        output += (
                f"{'# Edges':<20}" +
                "".join([f"{self.pyg_graphs[i].num_edges:>15,}" for i in range(network_count)]) + "\n"
        )
        # Node Attributes Dimension per graph
        output += (
                f"{'# Node Attributes':<20}" +
                "".join([
                    f"{self.pyg_graphs[i].num_node_features:>15,}"
                    for i in range(network_count)
                ]) + "\n"
        )
        # Edge Attributes Dimension per graph
        output += (
                f"{'# Edge Attributes':<20}" +
                "".join([
                    f"{self.pyg_graphs[i].num_edge_features:>15,}"
                    for i in range(network_count)
                ]) + "\n"
        )
        # Anchor Links (Train/Test)
        output += (
            f"{'=' * 60}\n"
            f"{'Anchor Links':<20}{'Train':>15}{'Test':>15}\n"
            f"{'-' * 60}\n"
            f"{f'Count (train ratio): {self.train_ratio})':<20}{self.train_data.shape[0]:>15,}{self.test_data.shape[0]:>15,}\n"
            f"{'=' * 60}"
        )
        return output


if __name__ == '__main__':
    for f in os.listdir('../../datasets/pyg'):
        if f.endswith('.pt'):
            dataset_name = f[:-3]
            dataset = Dataset(root='../../datasets/pyg', name=dataset_name, train_ratio=0.2, precision=64)
            print(dataset)
