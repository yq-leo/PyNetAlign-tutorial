import torch


def degree_normalize_sparse_tensor(edge_index, edge_weight, shape):
    """degree_normalize_sparse_tensor.
    """

    row, col = edge_index
    num_nodes = shape[0]
    deg = torch.zeros(num_nodes, dtype=edge_weight.dtype, device=edge_weight.device)
    deg = deg.scatter_reduce(0, row, edge_weight, reduce="sum", include_self=False)
    # deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-1)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    edge_weight = deg_inv_sqrt[row] * edge_weight

    return edge_index, edge_weight


def sparse_softmax(edge_index, edge_weight, shape):

    row, col = edge_index
    num_nodes = shape[0]
    edge_weight = torch.exp(edge_weight)
    deg = torch.zeros(num_nodes, dtype=edge_weight.dtype, device=edge_weight.device)
    deg = deg.scatter_reduce(0, row, edge_weight, reduce="sum", include_self=False)
    # deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes) + 1e-16
    deg_inv_sqrt = deg.pow(-1)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    edge_weight = deg_inv_sqrt[row] * edge_weight

    return edge_index, edge_weight


def sample_gumbel(shape, device, eps=1e-20):
    U = torch.rand(shape)
    U = U.to(device)
    return -torch.log(-torch.log(U + eps) + eps)


def sparse_gumbel_softmax(src, index, temperature, shape):
    y = src + sample_gumbel(len(src), src.device)
    edge_index, edge_weight = sparse_softmax(index, y / temperature, shape)
    return edge_index, edge_weight
