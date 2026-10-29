"""Pitfall guards: pairs, splits, scaler, label-independence of features, BM25 sanity."""
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler
import pairs as P
import data as D


def toy(n_q=6, n=12, seed=0):
    rng = np.random.default_rng(seed)
    qids = np.repeat([f"q{i}" for i in range(n_q)], n)
    y = rng.integers(0, 3, size=n_q * n)
    X = rng.normal(size=(n_q * n, 4))
    rank = np.tile(np.arange(1, n + 1), n_q).astype(float)
    return X, y, qids, rank


@pytest.mark.parametrize("mode,cap", [("plain", None), ("plain", 5), ("gap", None), ("dndcg", 7)])
def test_pairs_same_query_different_label_balanced(mode, cap):
    X, y, qids, rank = toy()
    Xd, yd, w, (I, J) = P.build_pairs(X, y, qids, rank, mode, np.random.default_rng(1), cap)
    assert np.all(qids[I] == qids[J])                     # never across queries
    assert np.all(y[I] != y[J])                           # never equal labels
    assert abs((yd == 1).sum() - (yd == -1).sum()) <= 1   # directions balanced
    sign = yd[:, None]
    assert np.allclose(Xd, sign * (X[I] - X[J]))          # +1 = (better - worse), -1 = flipped
    if cap:
        for q in np.unique(qids):
            assert (qids[I] == q).sum() <= cap
    if w is not None:
        assert np.all(w > 0)


def test_all_pairs_count_matches_formula():
    X, y, qids, rank = toy()
    expected = sum(int((y[qids == q][:, None] > y[qids == q][None, :]).sum()) for q in np.unique(qids))
    assert P.count_pairs(y, qids) == expected
    Xd, yd, *_ = P.build_pairs(X, y, qids, rank, "plain", np.random.default_rng(0))
    assert len(yd) == expected


def test_gap_weight_equals_grade_difference():
    X, y, qids, rank = toy()
    _, _, w, (I, J) = P.build_pairs(X, y, qids, rank, "gap", np.random.default_rng(0))
    assert np.array_equal(w, (y[I] - y[J]).astype(float))


def test_no_query_in_two_splits_detects_leak():
    ok = {"train": {"a": {}}, "dev": {"b": {}}, "test": {"c": {}}}
    D.assert_query_disjoint(ok)
    with pytest.raises(AssertionError):
        D.assert_query_disjoint({"train": {"a": {}}, "dev": {"a": {}}, "test": {"c": {}}})


def test_scaler_statistics_come_from_train_only():
    rng = np.random.default_rng(0)
    tr = rng.normal(5, 2, size=(200, 3)); te = rng.normal(-50, 9, size=(50, 3))
    sc = StandardScaler().fit(tr)
    assert np.allclose(sc.mean_, tr.mean(axis=0)) and not np.allclose(sc.mean_, np.vstack([tr, te]).mean(axis=0))


def test_saved_scaler_matches_train_split_if_results_exist():
    from config import RESULTS_DIR, PROC_DIR
    f, s = PROC_DIR / "full" / "features.pkl", RESULTS_DIR / "scaler_stats.csv"
    if not (f.exists() and s.exists()):
        pytest.skip("run the pipeline first")
    from features import FEATURE_NAMES
    df = pd.read_pickle(f); tr = df[df["split"] == "train"][FEATURE_NAMES]
    st = pd.read_csv(s)
    assert np.allclose(st["train_mean"], tr.mean().values) and np.allclose(st["train_std"], tr.std(ddof=0).values)
    # and the splits saved in the feature table are query-disjoint
    sets = {k: set(g["qid"]) for k, g in df.groupby("split")}
    assert not (sets["train"] & sets["dev"] or sets["train"] & sets["test"] or sets["dev"] & sets["test"])


def test_features_do_not_depend_on_labels_and_have_no_label_argument():
    import inspect
    from features import stage1_and_features, TextIndex, stage1_and_features as f
    assert not any("label" in p or "qrel" in p for p in inspect.signature(f).parameters)
    titles = ["breast cancer diet", "heart disease statin", "diet and heart health"]
    texts = ["cancer risk and diet study", "statin drugs lower heart risk", "healthy diet protects the heart"]
    idx = TextIndex(titles, texts, 1.2, 0.75)
    a = stage1_and_features(idx, ["q"], ["diet heart"], 100, 1000.0)
    b = stage1_and_features(idx, ["q"], ["diet heart"], 100, 1000.0)
    pd.testing.assert_frame_equal(a, b)


def test_bm25_hand_computed():
    from bm25 import count_matrix, bm25_idf, bm25_weight_matrix, binary_query_matrix
    docs = [["a", "b"], ["a"], ["c"]]
    vocab = {"a": 0, "b": 1, "c": 2}
    tf = count_matrix(docs, vocab)
    W = bm25_weight_matrix(tf, 1.2, 0.75)
    idf_a = np.log(1 + (3 - 2 + .5) / (2 + .5))
    avgdl = (2 + 1 + 1) / 3
    s0 = idf_a * 1 * 2.2 / (1 + 1.2 * (1 - .75 + .75 * 2 / avgdl))     # doc 0, term a
    Q = binary_query_matrix([["a", "zzz"]], vocab)
    got = (Q @ W.T).toarray()[0]
    assert got[0] == pytest.approx(s0) and got[2] == 0.0 and got[1] > got[0]   # shorter doc scores higher
