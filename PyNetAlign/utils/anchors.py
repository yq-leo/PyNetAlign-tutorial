import torch


def get_anchor_pairs(anchor_links, gid1, gid2):
    potential_pairs = anchor_links[:, [gid1, gid2]]
    anchor_pairs = potential_pairs[torch.all(potential_pairs != -1, dim=1)]
    return anchor_pairs


def get_pairwise_anchor_pairs(anchor_links):
    num_graphs = anchor_links.shape[1]
    anchor_pairs_dict = {}
    for gid1 in range(num_graphs):
        for gid2 in range(gid1 + 1, num_graphs):
            anchor_pairs = get_anchor_pairs(anchor_links, gid1, gid2)
            anchor_pairs_dict[(gid1, gid2)] = anchor_pairs
    return anchor_pairs_dict
