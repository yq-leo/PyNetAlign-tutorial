import torch


def get_anchor_pairs(gid1, gid2, anchor_links):
    potential_pairs = anchor_links[:, [gid1, gid2]]
    anchor_pairs = potential_pairs[torch.all(potential_pairs != -1, dim=1)]
    return anchor_pairs
