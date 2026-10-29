"""Tests for the extension code: hinge SVM, CV folds / inner split / scaler, VIF pruning, runner helpers."""
import numpy as np
import pandas as pd
import pytest
import models as MD
import ltr_runner as LR
import run_cv as CV
import pairs as P


# ---------------------------------------------------------------- hinge Ranking SVM
def _spy(monkeypatch):
    seen = {}
    real = MD.LinearSVC
    monkeypatch.setattr(MD, "LinearSVC", lambda **k: (seen.update(k), real(**k))[1])
    return seen


def test_hinge_svm_is_standard_hinge_dual_no_intercept(monkeypatch):
    seen = _spy(monkeypatch)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 4)); y = np.where(X[:, 0] + 0.1 * rng.normal(size=400) > 0, 1, -1)
    m = MD.fit_ranksvm_hinge(X, y, None, 1.0)
    assert seen["loss"] == "hinge" and seen["dual"] is True and seen["fit_intercept"] is False
    assert m["b"] == 0.0 and m["w"][0] > 0 and abs(m["w"][0]) > 3 * abs(m["w"][1:]).max()


def test_squared_hinge_svm_unchanged(monkeypatch):
    seen = _spy(monkeypatch)
    X = np.random.default_rng(1).normal(size=(100, 3)); y = np.where(X[:, 0] > 0, 1, -1)
    MD.fit_ranksvm(X, y, None, 1.0)
    assert seen["loss"] == "squared_hinge" and seen["dual"] is False and seen["fit_intercept"] is False


def test_hinge_variants_registered_with_distinct_seeds_and_same_pair_rules():
    assert LR.VARIANTS["ranksvm_hinge_all"][2] == "hinge" and LR.VARIANTS["ranksvm_all"][2] == "sq"
    assert len({v[3] for v in LR.VARIANTS.values()}) == len(LR.VARIANTS)
    # first four offsets must equal run_models.py (variant index) so that v1 results are reproduced
    assert [LR.VARIANTS[n][3] for n in ("ranksvm_all", "ranksvm_sampled", "ranksvm_gapweighted", "ranksvm_dndcgweighted_extra")] == [0, 1, 2, 3]


def test_runner_pairs_are_within_query_and_balanced():
    rng = np.random.default_rng(0)
    q = np.repeat([f"q{i}" for i in range(8)], 15); y = rng.integers(0, 2, size=len(q))
    X = rng.normal(size=(len(q), 3)); rk = np.tile(np.arange(1, 16), 8).astype(float)
    for name in LR.VARIANTS:
        Xd, yd, w, (I, J) = LR.make_pairs(name, X, y, q, rk, 5)
        assert np.all(q[I] == q[J]) and np.all(y[I] > y[J]) and abs((yd == 1).sum() - (yd == -1).sum()) <= 1


# ---------------------------------------------------------------- CV protocol
def test_folds_are_query_level_complete_and_balanced():
    qids = [f"q{i}" for i in range(103)]
    fold = CV.assign_folds(qids, 5, 42)
    assert set(fold) == set(qids) and set(fold.values()) == set(range(5))
    sizes = np.bincount(list(fold.values()))
    assert sizes.max() - sizes.min() <= 1
    assert CV.assign_folds(qids, 5, 42) == fold                 # deterministic
    assert CV.assign_folds(qids, 5, 43) != fold                 # seed matters


def test_row_level_split_would_be_detected_but_query_folds_never_share_queries():
    # rows of one query always carry the same fold (fold is a function of qid)
    qids = [f"q{i}" for i in range(40)]
    fold = CV.assign_folds(qids, 5, 1)
    rows = pd.DataFrame({"qid": np.repeat(qids, 7)}); rows["fold"] = rows["qid"].map(fold)
    assert (rows.groupby("qid")["fold"].nunique() == 1).all()
    for k in range(5):
        test_q = set(rows[rows.fold == k]["qid"]); train_q = set(rows[rows.fold != k]["qid"])
        assert not (test_q & train_q)


def test_inner_validation_split_is_query_level_disjoint_and_sized():
    pool = {f"q{i}" for i in range(100)}
    tr, va = CV.inner_split(pool, 0.2, 7)
    assert not (tr & va) and (tr | va) == pool and len(va) == 20
    assert CV.inner_split(pool, 0.2, 7) == (tr, va)


def test_scaler_helper_uses_training_frame_only():
    rng = np.random.default_rng(0)
    tr = pd.DataFrame(rng.normal(5, 2, size=(300, 2)), columns=["a", "b"])
    te = pd.DataFrame(rng.normal(-40, 9, size=(60, 2)), columns=["a", "b"])
    sc, (Xtr, Xte) = LR.scale(tr, [tr, te], ["a", "b"])
    assert np.allclose(sc.mean_, tr.mean().values) and np.allclose(Xtr.mean(axis=0), 0, atol=1e-9)
    assert Xte.mean() < -3                                       # test is NOT re-centred
    assert not np.allclose(sc.mean_, pd.concat([tr, te]).mean().values)


# ---------------------------------------------------------------- VIF pruning
def test_vif_detects_collinearity_and_pruning_reaches_threshold():
    rng = np.random.default_rng(0)
    a = rng.normal(size=2000); b = rng.normal(size=2000)
    df = pd.DataFrame({"a": a, "a_copy": a + 0.01 * rng.normal(size=2000), "b": b, "c": rng.normal(size=2000),
                       "ab": a + b + 0.05 * rng.normal(size=2000)})
    v = LR.vif(df)
    assert v["a"] > 100 and v["c"] < 1.5
    kept, drops = LR.vif_prune(df, 10.0)
    assert LR.vif(df[kept]).max() < 10.0
    assert len(drops) >= 2 and "c" in kept and (drops["vif_at_drop"] >= 10.0).all()   # only features above the threshold are dropped
    assert kept == [c for c in df.columns if c in kept]           # original order preserved


def test_vif_prune_keeps_everything_when_already_below_threshold():
    rng = np.random.default_rng(1)
    df = pd.DataFrame(rng.normal(size=(500, 4)), columns=list("abcd"))
    kept, drops = LR.vif_prune(df, 10.0)
    assert kept == list("abcd") and drops.empty


def test_saved_vif_pruning_matches_threshold_if_results_exist():
    from config import RESULTS_DIR
    f = RESULTS_DIR / "ext_vif_by_set.csv"
    if not f.exists():
        pytest.skip("run the extension pipeline first")
    v = pd.read_csv(f)
    assert v[v.feature_set == "v1_reduced"]["VIF"].max() < 10.0
