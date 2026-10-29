"""Metrics unit-tested on tiny hand-computed examples."""
import math
import numpy as np
import pytest
import metrics as M


def test_dcg_exp_gain_hand_computed():
    # ranking rels [2,0,1]: (2^2-1)/log2(2) + 0 + (2^1-1)/log2(4) = 3 + 0.5
    assert M.dcg_at_k([2, 0, 1], 3, "exp") == pytest.approx(3.5)
    # linear gain: 2/1 + 0 + 1/2 = 2.5
    assert M.dcg_at_k([2, 0, 1], 3, "lin") == pytest.approx(2.5)


def test_ndcg_idcg_uses_full_qrels_not_candidates():
    # full qrels = {2,1,1}; ranking only contains [1,0]. IDCG@10 exp = 3 + 1/log2(3) + 1/log2(4)
    idcg = 3 + 1 / math.log2(3) + 1 / 2
    dcg = 1 / 1.0
    assert M.ndcg_at_k([1, 0], [2, 1, 1], 10) == pytest.approx(dcg / idcg)


def test_ndcg_perfect_and_zero():
    assert M.ndcg_at_k([2, 1, 0], [2, 1], 10) == pytest.approx(1.0)
    assert M.ndcg_at_k([0, 0], [1], 10) == 0.0
    assert M.ndcg_at_k([1], [], 10) == 0.0


def test_ndcg_cutoff():
    # k=1: only first doc counts. rels [0,2], full [2]: dcg@1 = 0
    assert M.ndcg_at_k([0, 2], [2], 1) == 0.0
    assert M.ndcg_at_k([0, 2], [2], 2) == pytest.approx((3 / math.log2(3)) / 3)


def test_average_precision_binarises_and_uses_full_recall_denominator():
    # ranked rel [1,0,2,0]: P@1=1, P@3=2/3 ; 3 relevant overall (one never retrieved)
    assert M.average_precision([1, 0, 2, 0], 3) == pytest.approx((1 + 2 / 3) / 3)
    assert M.average_precision([0, 0], 2) == 0.0
    assert M.average_precision([1], 0) == 0.0


def test_pairwise_counts():
    # labels 2,1,0 ; scores order correct for all 3 pairs
    assert M.pairwise_counts([3, 2, 1], [2, 1, 0]) == (3.0, 3, 0)
    # fully reversed
    assert M.pairwise_counts([1, 2, 3], [2, 1, 0]) == (0.0, 3, 0)
    # equal labels create no pair; tie counts half
    assert M.pairwise_counts([1, 1, 5], [1, 1, 0]) == (0.0, 2, 0)    # 2 valid pairs, both wrong
    assert M.pairwise_counts([3, 3], [1, 0]) == (0.5, 1, 1)
    assert M.pairwise_counts([1, 2], [1, 1])[1] == 0


def test_rank_order_tiebreak_by_original_rank():
    assert list(M.rank_order([1.0, 1.0, 2.0], [1, 2, 3])) == [2, 0, 1]


def test_evaluate_scores_counts_queries_without_candidates():
    qrels = {"a": {"d1": 1}, "b": {"d9": 2}}
    rows = M.evaluate_scores(np.array(["a", "a"]), np.array([1, 0]), np.array([1, 2]),
                             np.array([2.0, 1.0]), qrels)
    by = {r["qid"]: r for r in rows}
    assert by["a"]["ndcg10"] == pytest.approx(1.0) and by["b"]["ndcg10"] == 0.0 and by["b"]["n_cand"] == 0


def test_crosscheck_with_ranx_if_available():
    ranx = pytest.importorskip("ranx")
    qrels = {"q1": {"d1": 2, "d2": 1, "d5": 1}, "q2": {"d3": 1}}
    run = {"q1": {"d2": 3.0, "d4": 2.0, "d1": 1.0}, "q2": {"d3": 0.5, "d8": 0.9}}
    q = ranx.Qrels(qrels); r = ranx.Run(run)
    got = ranx.evaluate(q, r, ["ndcg@10", "ndcg_burges@10", "map"], make_comparable=True)
    def per(qid, gain):
        order = sorted(run[qid], key=lambda d: -run[qid][d])
        rels = [qrels[qid].get(d, 0) for d in order]
        full = list(qrels[qid].values())
        return M.ndcg_at_k(rels, full, 10, gain), M.average_precision(rels, len(full))
    lin = np.mean([per(x, "lin")[0] for x in qrels]); exp = np.mean([per(x, "exp")[0] for x in qrels])
    ap = np.mean([per(x, "exp")[1] for x in qrels])
    assert got["ndcg@10"] == pytest.approx(lin) and got["ndcg_burges@10"] == pytest.approx(exp)
    assert got["map"] == pytest.approx(ap)
