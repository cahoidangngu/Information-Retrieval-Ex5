"""Stemmed Stage-1 experiment: analyzer, backward compatibility, candidate superset property, recall/oracle helpers."""
import inspect
import numpy as np
import pandas as pd
import pytest
import stage1_eval as S1
import metrics as M
from bm25 import tokenize
from features import TextIndex, stage1_and_features


def test_stem_analyzer_stems_after_stopword_removal():
    assert S1.stem_analyzer("The diets of cancers") == ["diet", "cancer"]
    assert S1.stem_analyzer("Diets, DIET!") == ["diet", "diet"]
    # stop-words are removed BEFORE stemming (same list as v1), so none of them survives in any form
    assert all(t not in S1.stem_analyzer("the and of this") for t in ("the", "and", "of", "thi"))
    assert S1.stem_analyzer("running relational") == ["run", "relat"]


def test_default_analyzer_is_v1_tokenizer_and_signature_has_no_labels():
    assert inspect.signature(TextIndex.__init__).parameters["analyzer"].default is tokenize
    assert not any("label" in p or "qrel" in p for p in inspect.signature(stage1_and_features).parameters)


TITLES = ["low carb diets", "diet and health", "heart disease", "carbohydrate study"]
TEXTS = ["people eating a diet low in carbohydrates", "diets for healthy hearts", "diseases of the heart", "studying carbs"]


def test_stemming_changes_matching_and_candidates_are_a_superset():
    q = ["diets carbohydrates"]
    un = TextIndex(TITLES, TEXTS, 1.2, 0.75)
    st = TextIndex(TITLES, TEXTS, 1.2, 0.75, analyzer=S1.stem_analyzer)
    cu = set(stage1_and_features(un, ["q"], q, 100, 1000.0)["doc_idx"])
    cs = set(stage1_and_features(st, ["q"], q, 100, 1000.0)["doc_idx"])
    assert cu <= cs                      # a doc matching a surface term also matches its stem
    assert cs - cu                       # ... and stemming adds docs (doc 3 contains only the singular 'carbohydrate')
    assert len(st.vocab) < len(un.vocab)


def test_default_index_features_unchanged_by_analyzer_refactor():
    a = stage1_and_features(TextIndex(TITLES, TEXTS, 1.2, 0.75), ["q"], ["low carb diets"], 100, 1000.0)
    b = stage1_and_features(TextIndex(TITLES, TEXTS, 1.2, 0.75, analyzer=tokenize), ["q"], ["low carb diets"], 100, 1000.0)
    pd.testing.assert_frame_equal(a, b)


def test_recall_summary_hand_computed():
    cr = pd.DataFrame({"split": ["test"] * 4, "n_cand": [100, 40, 0, 100], "n_rel_total": [10, 4, 5, 2], "n_rel_in_cand": [5, 4, 0, 0]})
    r = S1.recall_summary(cr).iloc[0]
    assert r["recall@100_micro"] == pytest.approx(9 / 21)
    assert r["recall@100_macro"] == pytest.approx((0.5 + 1.0 + 0.0 + 0.0) / 4)
    assert r["frac_queries_no_relevant_candidate"] == 0.5 and r["n_queries_no_relevant_candidate"] == 2
    assert r["frac_queries_lt100_cand"] == 0.5


def test_oracle_dominates_stage1_and_counts_queries_without_candidates():
    qrels = {"a": {"d1": 1, "d2": 2}, "b": {"d9": 1}}          # query b has no candidate rows
    frame = pd.DataFrame({"qid": ["a", "a", "a"], "label": [0, 2, 1], "bm25_rank": [1.0, 2.0, 3.0], "bm25_comb": [3.0, 2.0, 1.0]})
    r = S1.oracle_and_stage1(frame, qrels)
    assert r["oracle_NDCG@10"] >= r["stage1_NDCG@10"] and r["oracle_MAP"] >= r["stage1_MAP"]
    # oracle ranks [2,1,0] -> NDCG 1.0 for query a; query b contributes 0 -> mean 0.5
    assert r["oracle_NDCG@10"] == pytest.approx(0.5)
    assert r["oracle_MAP"] == pytest.approx(0.5)


def test_saved_stem_results_consistent_if_present():
    from config import RESULTS_DIR
    f, g = RESULTS_DIR / "ext_stem_candidate_recall.csv", RESULTS_DIR / "ext_stem_stage1_summary.csv"
    if not (f.exists() and g.exists()):
        pytest.skip("run the stemmed-stage1 experiment first")
    cr = pd.read_csv(f); summ = pd.read_csv(g)
    mine = S1.recall_summary(cr).set_index("split")
    saved = summ[summ["stage1"] == "Porter-stemmed"].set_index("split")
    for s in ("train", "dev", "test"):
        assert saved.loc[s, "recall@100_micro"] == pytest.approx(mine.loc[s, "recall@100_micro"])
    # same evaluated queries as the unstemmed main run
    main = pd.read_csv(RESULTS_DIR / "candidate_recall.csv")
    assert set(cr["qid"]) == set(main["qid"])
    # stemming can only add candidate matches before the Top-100 cut -> no query loses all of its candidates
    m = main.set_index("qid"); c = cr.set_index("qid")
    assert ((c["n_cand"] > 0) | (m.loc[c.index, "n_cand"] == 0)).all()
