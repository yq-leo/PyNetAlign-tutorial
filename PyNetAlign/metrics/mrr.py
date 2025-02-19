import torch


def mrr_ltr_score(similarity, test_pairs):
    r"""Mean Reciprocal Rank (MRR) score of graph1(left) to graph2(right) alignment."""
    ranks1 = torch.argsort(-similarity[test_pairs[:, 0]], dim=1)
    signal1_hit = ranks1 == test_pairs[:, 1].view(-1, 1)
    mrr = torch.mean(1 / (torch.where(signal1_hit)[1].float() + 1))
    return mrr


def mrr_rtl_score(similarity, test_pairs):
    r"""Mean Reciprocal Rank (MRR) score of graph2(right) to graph1(left) alignment."""
    ranks2 = torch.argsort(-similarity.T[test_pairs[:, 1]], dim=1)
    signal2_hit = ranks2 == test_pairs[:, 0].view(-1, 1)
    mrr = torch.mean(1 / (torch.where(signal2_hit)[1].float() + 1))
    return mrr


def mrr_max_score(similarity, test_pairs):
    r"""Max Mean Reciprocal Rank (MRR) score of left-to-right and right-to-left alignments."""
    mrr_ltr = mrr_ltr_score(similarity, test_pairs)
    mrr_rtl = mrr_rtl_score(similarity, test_pairs)
    mrr = torch.max(mrr_ltr, mrr_rtl)

    return mrr


def mrr_mean_score(similarity, test_pairs):
    r"""Mean Mean Reciprocal Rank (MRR) score of left-to-right and right-to-left alignments."""
    mrr_ltr = mrr_ltr_score(similarity, test_pairs)
    mrr_rtl = mrr_rtl_score(similarity, test_pairs)
    mrr = (mrr_ltr + mrr_rtl) / 2

    return mrr
