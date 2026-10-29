"""Download + load BEIR NFCorpus, dataset statistics."""
import hashlib, json, urllib.request, zipfile, collections
import numpy as np
from config import RAW_DIR, DATA_DIR, DATA_URL

SPLITS = ["train", "dev", "test"]


def download():
    """Download and extract NFCorpus if missing. Returns sha256 of the zip (or None)."""
    z = RAW_DIR / "nfcorpus.zip"
    if (DATA_DIR / "corpus.jsonl").exists():
        return hashlib.sha256(z.read_bytes()).hexdigest() if z.exists() else None
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(DATA_URL, z)
    with zipfile.ZipFile(z) as f:
        f.extractall(RAW_DIR / "nfcorpus")
    return hashlib.sha256(z.read_bytes()).hexdigest()


def load():
    """Returns doc_ids, titles, texts, queries{qid:text}, qrels{split:{qid:{docid:rel}}}."""
    doc_ids, titles, texts = [], [], []
    with open(DATA_DIR / "corpus.jsonl", encoding="utf8") as f:
        for line in f:
            d = json.loads(line)
            doc_ids.append(d["_id"]); titles.append(d["title"]); texts.append(d["text"])
    queries = {}
    with open(DATA_DIR / "queries.jsonl", encoding="utf8") as f:
        for line in f:
            d = json.loads(line); queries[d["_id"]] = d["text"]
    qrels = {}
    for s in SPLITS:
        q = collections.defaultdict(dict)
        with open(DATA_DIR / "qrels" / f"{s}.tsv", encoding="utf8") as f:
            next(f)
            for line in f:
                qid, did, rel = line.rstrip("\n").split("\t")
                q[qid][did] = int(rel)
        qrels[s] = dict(q)
    return doc_ids, titles, texts, queries, qrels


def subset_qrels(qrels, n_per_split, seed):
    """Smoke-test helper: random query subset per split (query level, seeded)."""
    rng = np.random.default_rng(seed)
    out = {}
    for s, q in qrels.items():
        ids = sorted(q)
        keep = rng.choice(len(ids), size=min(n_per_split, len(ids)), replace=False)
        out[s] = {ids[i]: q[ids[i]] for i in sorted(keep)}
    return out


def assert_query_disjoint(qrels):
    """Pitfall guard: no query id may appear in two splits."""
    for a in SPLITS:
        for b in SPLITS:
            if a < b:
                inter = set(qrels[a]) & set(qrels[b])
                assert not inter, f"queries in both {a} and {b}: {sorted(inter)[:5]}"


def dataset_stats(doc_ids, titles, texts, queries, qrels):
    st = {"n_docs": len(doc_ids), "n_queries_total": len(queries)}
    wl = [len((t + " " + x).split()) for t, x in zip(titles, texts)]
    st["doc_words_mean"] = float(np.mean(wl)); st["doc_words_median"] = float(np.median(wl))
    groups = collections.Counter(t + "||" + x for t, x in zip(titles, texts))
    st["n_duplicate_text_groups"] = sum(1 for v in groups.values() if v > 1)
    st["n_docs_in_duplicate_groups"] = sum(v for v in groups.values() if v > 1)
    for s in SPLITS:
        q = qrels[s]
        lv = collections.Counter(r for d in q.values() for r in d.values())
        st[s] = {"n_queries": len(q), "n_judgments": sum(len(d) for d in q.values()),
                 "levels": {str(k): v for k, v in sorted(lv.items())},
                 "mean_judged_per_query": float(np.mean([len(d) for d in q.values()])),
                 "n_docs_judged": len({x for d in q.values() for x in d})}
    tr_docs = {x for d in qrels["train"].values() for x in d}
    for s in ["dev", "test"]:
        sd = {x for d in qrels[s].values() for x in d}
        st[f"{s}_judged_docs_also_judged_in_train"] = len(sd & tr_docs) / len(sd)
    return st
