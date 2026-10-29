"""Feature set v2: hand-computed checks, label independence, v1 untouched."""
import inspect
import numpy as np
import pandas as pd
import pytest
import features_v2 as F2
from features import FEATURE_NAMES


def test_v2_extends_v1_without_changing_it():
    assert F2.V2_NAMES[: len(FEATURE_NAMES)] == FEATURE_NAMES and len(F2.V2_NEW) == 5
    assert len(set(F2.V2_NAMES)) == len(F2.V2_NAMES)


def test_no_label_argument():
    assert not any("label" in p or "qrel" in p for p in inspect.signature(F2.compute_v2_new).parameters)


def test_proximity_hand_computed():
    # tokens: a x b x x c  -> a@0, b@2, c@5
    pos = {"a": np.array([0]), "b": np.array([2]), "c": np.array([5])}
    mn, avg = F2.proximity(pos, ["a", "b", "c"], 6)
    assert mn == 2.0 and avg == pytest.approx((2 + 5 + 3) / 3)       # d(a,b)=2, d(a,c)=5, d(b,c)=3
    # smallest distance picks the closest occurrences
    pos = {"a": np.array([0, 10]), "b": np.array([4, 11])}
    assert F2.proximity(pos, ["a", "b"], 20) == (1.0, 1.0)
    # fewer than 2 matched terms -> doc length (no proximity evidence)
    assert F2.proximity({"a": np.array([3])}, ["a", "zz"], 9) == (9.0, 9.0)


def test_bigrams():
    assert F2.bigrams(["a", "b", "c"]) == {("a", "b"), ("b", "c")}
    assert F2.bigrams(["a"]) == set()


def test_compute_v2_new_toy_corpus():
    titles = ["low carb diets", "carb counting", "unrelated topic"]
    texts = ["people eating diets low in carb", "counting carbs and diets", "nothing here"]
    v1 = pd.DataFrame({"qid": ["q", "q", "q"], "doc_idx": [0, 1, 2]})
    out = F2.compute_v2_new(v1, titles, texts, ["q"], ["low carb diets"])
    assert list(out.columns) == F2.V2_NEW and len(out) == 3
    # query bigrams: (low,carb), (carb,diets); doc0 title has both
    assert out.loc[0, "bigram_match_count"] == 2
    assert out.loc[1, "bigram_match_count"] == 0       # "carb counting" / "counting carbs and diets": no adjacent match
    assert out.loc[2, "bigram_match_count"] == 0
    # stemming lets "carbs"/"diets" match: doc1 stem-BM25 > 0 and > unstemmed-only evidence for 'carbs'
    assert out.loc[1, "bm25_stem_comb"] > 0 and out.loc[2, "bm25_stem_comb"] == 0
    # doc0 tokens: low carb diets people eating diets low in carb -> positions low@0,6 carb@1,8 diets@2,5 ; min dist 1
    assert out.loc[0, "min_term_dist"] == 1.0
    from bm25 import tokenize
    assert out.loc[2, "min_term_dist"] == len(tokenize("unrelated topic") + tokenize("nothing here"))   # no match -> doc length (after stop-word removal)
    assert (out["bm25_k2_b05_comb"].values >= 0).all()
