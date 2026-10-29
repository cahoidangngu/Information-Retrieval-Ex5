"""Cross-check our hand-written metrics against ranx (and pytrec_eval if installable) on the real test run.

Rankings are passed as strictly decreasing scores built from OUR ordering, so tie-handling cannot differ.
Usage: python -I src/crosscheck.py [--smoke]
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import data as D
import metrics as M
from util import dirs, Logger


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    _, res = dirs(args.smoke); log = Logger(res / "run_log.txt")
    _, _, _, _, qrels = D.load()
    cr = pd.read_csv(res / "candidate_recall.csv"); qs = set(cr.loc[cr["split"] == "test", "qid"])
    qr = {q: d for q, d in qrels["test"].items() if q in qs}
    ts = pd.read_csv(res / "test_scores.csv.gz"); pq = pd.read_csv(res / "per_query_test.csv")
    rows = []
    try:
        import ranx
    except ImportError:
        log("ranx not installed - cross-check skipped"); return
    try:
        import pytrec_eval
    except ImportError:
        pytrec_eval = None
    for model in ["bm25", "ranksvm_sampled", "logreg"]:
        run = {}
        for qid, g in ts.groupby("qid", sort=False):
            o = M.rank_order(g[model].values, g["bm25_rank"].values)
            run[qid] = {g["doc_id"].values[i]: float(len(o) - k) for k, i in enumerate(o)}
        for q in qs:
            run.setdefault(q, {})
        Q = ranx.Qrels(qr); R = ranx.Run({k: v for k, v in run.items() if v})
        got = ranx.evaluate(Q, R, ["ndcg@10", "ndcg_burges@10", "ndcg@5", "map@100"], make_comparable=True)
        mine = pq[pq["model"] == model]
        r = {"model": model, "metric_set": "ranx",
             "ndcg10_lin_ours": mine["ndcg10_lin"].mean(), "ndcg10_lin_ref": got["ndcg@10"],
             "ndcg10_exp_ours": mine["ndcg10"].mean(), "ndcg10_exp_ref": got["ndcg_burges@10"],
             "ndcg5_exp_ours": mine["ndcg5"].mean(), "map_ours": mine["ap"].mean(), "map_ref": got["map@100"]}
        r["max_abs_diff"] = max(abs(r["ndcg10_lin_ours"] - r["ndcg10_lin_ref"]), abs(r["ndcg10_exp_ours"] - r["ndcg10_exp_ref"]),
                                abs(r["map_ours"] - r["map_ref"]))
        rows.append(r)
        if pytrec_eval:
            ev = pytrec_eval.RelevanceEvaluator({q: {d: int(v) for d, v in x.items()} for q, x in qr.items()}, {"map_cut_100", "ndcg_cut_10"})
            out = ev.evaluate({q: v for q, v in run.items()})
            tre = {q: out.get(q, {"map_cut_100": 0.0, "ndcg_cut_10": 0.0}) for q in qs}
            rows[-1]["pytrec_ndcg10_lin"] = np.mean([v["ndcg_cut_10"] for v in tre.values()])
            rows[-1]["pytrec_map100"] = np.mean([v["map_cut_100"] for v in tre.values()])
    out = pd.DataFrame(rows); out.to_csv(res / "crosscheck_metrics.csv", index=False)
    log("crosscheck vs ranx:\n" + out.to_string())
    assert out["max_abs_diff"].max() < 1e-9, "our metrics disagree with ranx"


if __name__ == "__main__":
    main()
