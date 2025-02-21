from typing import Optional
import time
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import degree

from PyNetAlign.data import Dataset
from PyNetAlign.utils import get_anchor_pairs
from .base_model import BaseModel


class IONE(BaseModel):
    r"""IONE algorithm for pairwise plain network alignment"""

    def __init__(self,
                 dataset: Dataset,
                 gid1: int,
                 gid2: int,
                 total_epochs: int,
                 out_dim: Optional[int] = 100,
                 precision: Optional[int] = 32):
        super(IONE, self).__init__(precision=precision)

        assert isinstance(dataset, Dataset), 'Input dataset must be a PyNetAlign Dataset object'
        assert 0 <= gid1 < len(dataset.pyg_graphs) and 0 <= gid2 < len(dataset.pyg_graphs), 'Invalid graph IDs'
        assert gid1 != gid2, 'Cannot align a graph with itself'
        assert out_dim > 0, 'Output dimension must be a positive integer'
        assert total_epochs > 0, 'Total epochs must be a positive integer'

        graph1, graph2 = dataset.pyg_graphs[gid1], dataset.pyg_graphs[gid2]
        self.two_order_x = IONEUpdate(graph1, out_dim, precision)
        self.two_order_y = IONEUpdate(graph2, out_dim, precision)

        anchors = get_anchor_pairs(dataset.train_data, gid1, gid2)
        self.anchor_map1 = {anchor[0].item(): anchor[1].item() for anchor in anchors.cpu()}
        self.anchor_map2 = {anchor[1].item(): anchor[0].item() for anchor in anchors.cpu()}
        # self.anchor_map1 = {f'{anchor[0]}_{gid1}': f'{anchor[1]}_{gid2}' for anchor in anchors}
        # self.anchor_map2 = {f'{anchor[1]}_{gid2}': f'{anchor[0]}_{gid1}' for anchor in anchors}

        self.epochs_cnt = 0
        self.total_epochs = total_epochs

    def forward(self):
        self.two_order_x(i=self.epochs_cnt,
                         iter_count=self.total_epochs,
                         two_order_embeddings=self.two_order_x.embeddings,
                         two_order_emb_context_input=self.two_order_x.emb_context_input,
                         two_order_emb_context_output=self.two_order_x.emb_context_output,
                         anchors=self.anchor_map1,
                         same_network=True)
        self.two_order_y(i=self.epochs_cnt,
                         iter_count=self.total_epochs,
                         two_order_embeddings=self.two_order_x.embeddings,
                         two_order_emb_context_input=self.two_order_x.emb_context_input,
                         two_order_emb_context_output=self.two_order_x.emb_context_output,
                         anchors=self.anchor_map2,
                         same_network=False)
        self.epochs_cnt += 1

        emb_x = F.normalize(self.two_order_x.embeddings, p=2, dim=1)
        emb_y = F.normalize(self.two_order_y.embeddings, p=2, dim=1)

        return emb_x, emb_y

    def to(self, device):
        self.two_order_x = self.two_order_x.to(device)
        self.two_order_y = self.two_order_y.to(device)
        return self


class IONEUpdate(BaseModel):
    def __init__(self,
                 graph: Data,
                 out_dim: int,
                 precision: Optional[int] = 32):
        super(IONEUpdate, self).__init__(precision=precision)

        self.graph = graph
        self.dimension = out_dim

        self.embeddings = torch.empty((self.graph.num_nodes, self.dimension), dtype=self.precision).uniform_(
            -0.5 / self.dimension, 0.5 / self.dimension)
        self.emb_context_input = torch.zeros(self.graph.num_nodes, self.dimension, dtype=self.precision)
        self.emb_context_output = torch.zeros(self.graph.num_nodes, self.dimension, dtype=self.precision)
        self.vertex = (degree(self.graph.edge_index[0], num_nodes=self.graph.num_nodes) +
                       degree(self.graph.edge_index[1], num_nodes=self.graph.num_nodes))

        self.init_rho = 0.025
        self.rho = 0
        self.num_negative = 5
        self.neg_table_size = 10000000

        self.edge_weight = []
        self.prob = torch.zeros(self.graph.num_edges, dtype=self.precision)
        self.alias = torch.zeros(self.graph.num_edges, dtype=torch.int64)
        self.neg_table = torch.zeros(self.neg_table_size, dtype=torch.int64)

        # Initialize tables
        start = time.time()
        self.init_alias_table()
        print(f'{self.graph.name}: alias table initialized in {time.time() - start:.2f} seconds')
        start = time.time()
        self.init_neg_table()
        print(f'{self.graph.name}: negative table initialized in {time.time() - start:.2f} seconds')

    def init_alias_table(self):
        # TODO: Incroporate edge weights
        self.edge_weight = torch.ones(self.graph.num_edges, dtype=self.precision)
        norm_prob = F.normalize(self.edge_weight, p=1, dim=0) * self.graph.num_edges

        small_block = torch.flip(torch.argwhere(norm_prob < 1).flatten(), dims=[0])
        large_block = torch.flip(torch.argwhere(norm_prob >= 1).flatten(), dims=[0])

        num_small_block = len(small_block)
        num_large_block = len(large_block)
        while num_small_block > 0 and num_large_block > 0:
            num_small_block = num_small_block - 1
            cur_small_block = small_block[num_small_block]
            num_large_block = num_large_block - 1
            cur_large_block = large_block[num_large_block]

            self.prob[cur_small_block] = norm_prob[cur_small_block]
            self.alias[cur_small_block] = cur_large_block

            norm_prob[cur_large_block] = norm_prob[cur_large_block] + norm_prob[cur_small_block] - 1
            if norm_prob[cur_large_block] < 1:
                small_block[num_small_block] = cur_large_block
                num_small_block = num_small_block + 1
            else:
                large_block[num_large_block] = cur_large_block
                num_large_block = num_large_block + 1

        while num_large_block > 0:
            num_large_block = num_large_block - 1
            self.prob[large_block[num_large_block]] = 1

        while num_small_block > 0:
            num_small_block = num_small_block - 1
            self.prob[small_block[num_small_block]] = 1

    def sample_edge(self, rand1: float, rand2: float) -> int:
        k = int(len(self.edge_weight) * rand1)
        return k if rand2 < self.prob[k] else self.alias[k]

    def init_neg_table(self):
        total_sum = torch.sum(self.vertex ** 0.75).cpu().item()

        cumulative_sum = 0
        por = 0
        perm_node_list = torch.randperm(self.graph.num_nodes).numpy().tolist()
        list_iter = iter(perm_node_list)
        current = next(list_iter)

        vertex = self.vertex.cpu().numpy().tolist()
        self.neg_table = []
        for i in range(self.neg_table_size):
            if (i + 1) / self.neg_table_size > por:
                cumulative_sum += vertex[current] ** 0.75
                por = cumulative_sum / total_sum
                if por >= 1:
                    self.neg_table.append(current)
                    continue
                if i != 0:
                    current = next(list_iter)
            self.neg_table.append(current)
        self.neg_table = torch.tensor(self.neg_table, dtype=torch.int64).to(self.vertex.device)

    def update(self, vec_u, vec_v, label, source, target, two_order_embeddings, two_order_emb_context,
               anchors, same_network=True):
        if source in anchors:
            vec_u = two_order_embeddings[anchors[source]] if not same_network else two_order_embeddings[source]
        if target in anchors:
            vec_v = two_order_emb_context[anchors[target]] if not same_network else two_order_emb_context[target]
            # vec_v = two_order_emb_context[anchors[target]] if anchors[target] in two_order_emb_context else two_order_emb_context[target]

        x = vec_u @ vec_v
        g = (label - torch.sigmoid(x)) * self.rho

        vec_error = g * vec_v
        if target in anchors:
            if same_network:
                vec_v += g * vec_u
            else:
                two_order_emb_context[anchors[target]] += g * vec_u
        else:
            vec_v += g * vec_u

        return vec_error

    def update_reverse(self, vec_u, vec_v, label, source, target, two_order_embeddings, two_order_emb_context,
                       anchors, same_network=True):
        if source in anchors:
            vec_u = two_order_embeddings[anchors[source]] if not same_network else two_order_embeddings[source]
        if target in anchors:
            vec_v = two_order_emb_context[anchors[target]] if not same_network else two_order_emb_context[target]

        x = vec_u @ vec_v
        g = (label - torch.sigmoid(x)) * self.rho

        vec_error = g * vec_v
        if target in anchors:
            if same_network:
                vec_v += g * vec_u
            else:
                two_order_emb_context[anchors[target]] += g * vec_u
        else:
            vec_v += g * vec_u

        uid_1 = source
        if uid_1 in anchors:
            if same_network:
                self.embeddings[uid_1] += vec_error
            else:
                two_order_embeddings[anchors[uid_1]] += vec_error
        else:
            self.embeddings[uid_1] += vec_error

    def forward(self, i, iter_count, two_order_embeddings, two_order_emb_context_input, two_order_emb_context_output,
                anchors, same_network=True):
        vec_error = torch.zeros(self.dimension, dtype=self.precision)
        if i % int(iter_count / 10) == 0:
            self.rho = self.init_rho * (1.0 - i / iter_count)
            if self.rho < self.init_rho * 0.0001:
                self.rho = self.init_rho * 0.0001

        edge_id = self.sample_edge(torch.rand(1).item(), torch.rand(1).item())
        uid_1, uid_2 = self.graph.edge_index[:, edge_id].cpu()
        uid_1, uid_2 = uid_1.item(), uid_2.item()

        d = 0
        while d < self.num_negative + 1:
            if d == 0:
                label = 1
                target = uid_2
            else:
                neg_index = torch.randint(0, self.neg_table_size, (1,)).item()
                target = self.neg_table[neg_index].cpu().item()
                assert not isinstance(target, torch.Tensor), 'Target should not be a tensor'
                if target == uid_1 or target == uid_2:
                    continue
                label = 0

            vec_error += self.update(vec_u=self.embeddings[uid_1],
                                     vec_v=self.emb_context_input[target],
                                     label=label,
                                     source=uid_1,
                                     target=target,
                                     two_order_embeddings=two_order_embeddings,
                                     two_order_emb_context=two_order_emb_context_input,
                                     anchors=anchors,
                                     same_network=same_network)
            self.update_reverse(vec_u=self.embeddings[target],
                                vec_v=self.emb_context_output[uid_1],
                                label=label,
                                source=target,
                                target=uid_1,
                                two_order_embeddings=two_order_embeddings,
                                two_order_emb_context=two_order_emb_context_output,
                                anchors=anchors,
                                same_network=same_network)
            d = d + 1

        if uid_1 in anchors:
            vec_u = two_order_embeddings[anchors[uid_1]] if not same_network else two_order_embeddings[uid_1]
            if vec_u is None:
                self.embeddings[uid_1] += vec_error
            else:
                two_order_embeddings[anchors[uid_1]] += vec_error

        else:
            self.embeddings[uid_1] += vec_error

    def to(self, device):
        self.embeddings = self.embeddings.to(device)
        self.emb_context_input = self.emb_context_input.to(device)
        self.emb_context_output = self.emb_context_output.to(device)
        self.vertex = self.vertex.to(device)
        self.prob = self.prob.to(device)
        self.alias = self.alias.to(device)
        self.neg_table = self.neg_table.to(device)
        return self
