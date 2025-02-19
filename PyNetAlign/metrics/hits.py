import torch


def hits_ks_ltr_scores(similarity, test_pairs, ks=None):
    r"""Hits@K scores of graph1(left) to graph2(right) alignment."""
    hits_ks = {}
    ranks1 = torch.argsort(-similarity[test_pairs[:, 0]], dim=1)
    signal1_hit = ranks1 == test_pairs[:, 1].view(-1, 1)
    for k in ks:
        hits_ks[k] = torch.sum(signal1_hit[:, :k]) / test_pairs.shape[0]

    return hits_ks


def hits_ks_rtl_scores(similarity, test_pairs, ks=None):
    r"""Hits@K scores of graph2(right) to graph1(left) alignment."""
    hits_ks = {}
    ranks2 = torch.argsort(-similarity.T[test_pairs[:, 1]], dim=1)
    signal2_hit = ranks2 == test_pairs[:, 0].view(-1, 1)
    for k in ks:
        hits_ks[k] = torch.sum(signal2_hit[:, :k]) / test_pairs.shape[0]

    return hits_ks


def hits_ks_max_scores(similarity, test_pairs, ks=None):
    r"""Max Hits@K scores of left-to-right and right-to-left alignments."""
    hits_ks = {}

    hits_ks_ltr = hits_ks_ltr_scores(similarity, test_pairs, ks=ks)
    hits_ks_rtl = hits_ks_rtl_scores(similarity, test_pairs, ks=ks)
    for k in ks:
        hits_ks[k] = torch.max(hits_ks_ltr[k], hits_ks_rtl[k])

    return hits_ks


def hits_ks_mean_scores(similarity, test_pairs, ks=None):
    r"""Mean Hits@K scores of left-to-right and right-to-left alignments."""
    hits_ks = {}

    hits_ks_ltr = hits_ks_ltr_scores(similarity, test_pairs, ks=ks)
    hits_ks_rtl = hits_ks_rtl_scores(similarity, test_pairs, ks=ks)
    for k in ks:
        hits_ks[k] = (hits_ks_ltr[k] + hits_ks_rtl[k]) / 2

    return hits_ks
