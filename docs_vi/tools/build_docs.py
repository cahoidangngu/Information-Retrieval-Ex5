"""Builds docs_vi/*.md from docs_vi/templates/*.tmpl.md.

Every number in the generated documents that is attributed to a project file is produced by a placeholder, so it is READ from
that file at build time (never typed by hand). Placeholder syntax:  [[kind|arg|arg|...]]

  [[csv|file|k=v&k2=v2|column|fmt]]                  one cell of a CSV (row chosen by the selectors; file relative to results/)
  [[table|file|keycol|key1;key2|col1;col2|fmt|key1=Name1;key2=Name2|filter]]   markdown table of cells (filter e.g. model=ranksvm_all)
  [[count|file|k=v&k2=v2|fmt]]                       number of CSV rows matching the selectors (empty = all rows)
  [[json|file|dotted.path|fmt]]                       one value of a JSON file
  [[txt|file|expr|fmt]]                               first number of a text file, transformed by expr in x (e.g. x/60)
  [[cfg|NAME|fmt]]                                    a constant of src/config.py
  [[calc|expr|fmt]]                                   arithmetic (hand-computable toy examples); functions: log2, sqrt
  [[toy|name|fmt]]                                    a toy metric computed with the PROJECT's own src/metrics.py
A file name starting with "docs_vi/" is relative to the project root, otherwise relative to results/.
fmt is a Python format spec (".4f", ",d", ".1%", ...) or "s" for the raw text of the file.

Also writes docs_vi/number_manifest.json (every resolved value with its source) and the data_vi/*.csv extracts of the two
worked queries (derived from results/test_scores.csv.gz + data/processed/full/features.pkl + the corpus; verified by the checker).
Usage:  python -I docs_vi/tools/build_docs.py
"""
import ast, csv, json, math, operator, re, sys
sys.dont_write_bytecode = True
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
DOCS = ROOT / "docs_vi"
sys.path.insert(0, str(ROOT / "src"))

EXAMPLE_QUERIES = {"PLAIN-1537": "q1537", "PLAIN-3085": "q3085"}     # improved / harmed (see results/query_evidence.csv)
PAIR_QID = "PLAIN-1537"
PAIR_DOCS = {"relevant": "MED-2588", "bm25_top1_not_relevant": "MED-4880"}


# ----------------------------------------------------------------------------- resolvers
def path_of(f):
    return ROOT / f if f.startswith("docs_vi/") else RES / f


def read_csv(f):
    with open(path_of(f), encoding="utf8", newline="") as fh:
        return list(csv.DictReader(fh))


COND = re.compile(r"^([^<>=&]+?)(<=|>=|<|>|=)(.*)$")


def match(row, sel):
    """All conditions of a selector must hold: col=text (string equality) or col<num, col>num, col<=num, col>=num."""
    for c in (sel.split("&") if sel else []):
        k, op, v = COND.match(c).groups()
        if op == "=":
            if row[k] != v:
                return False
        else:
            x, y = float(row[k]), float(v)
            if not {"<": x < y, ">": x > y, "<=": x <= y, ">=": x >= y}[op]:
                return False
    return True


def pick(rows, sel):
    hit = [r for r in rows if match(r, sel)]
    if len(hit) != 1:
        raise ValueError(f"selector {sel!r} matched {len(hit)} rows")
    return hit[0]


def fmtnum(raw, fmt):
    if fmt == "s":
        return raw
    v = float(raw)
    if fmt.endswith("d"):
        return format(int(round(v)), fmt)
    return format(v, fmt)


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.Pow: operator.pow,
        ast.USub: operator.neg}


def safe_eval(expr, names=None):
    names = names or {}

    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        if isinstance(n, ast.Name) and n.id in names:
            return names[n.id]
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ("log2", "sqrt") and len(n.args) == 1:
            return getattr(math, n.func.id)(ev(n.args[0]))
        raise ValueError(f"unsupported expression: {expr!r}")
    return ev(ast.parse(expr, mode="eval").body)


def toy_values():
    import metrics as M
    c, t, _ = M.pairwise_counts([3, 1, 2], [2, 1, 0])
    return {"ap_example": M.average_precision([1, 0, 1, 0], 3),
            "ndcg_example": M.ndcg_at_k([2, 0, 1], [2, 1, 0], 10),
            "dcg_example": M.dcg_at_k([2, 0, 1], 3),
            "idcg_example": M.dcg_at_k([2, 1, 0], 3),
            "pairacc_example": c / t}


MANIFEST = []
L, Rr = chr(0xE000), chr(0xE001)          # private-use markers around every file-derived number (tagged copy only)
W = lambda v: L + v + Rr


def resolve(kind, args, doc):
    if kind == "csv":
        f, sel, col, fmt = args
        raw = pick(read_csv(f), sel)[col]
        out = fmtnum(raw, fmt)
        MANIFEST.append({"doc": doc, "kind": kind, "args": args, "value": out})
        return W(out)
    if kind == "count":
        f, sel, fmt = args
        out = format(len([r for r in read_csv(f) if match(r, sel)]), fmt)
        MANIFEST.append({"doc": doc, "kind": kind, "args": args, "value": out})
        return W(out)
    if kind == "json":
        f, path, fmt = args
        v = json.loads(path_of(f).read_text(encoding="utf8"))
        for p in path.split("."):
            v = v[p]
        out = fmtnum(str(v), fmt)
        MANIFEST.append({"doc": doc, "kind": kind, "args": args, "value": out})
        return W(out)
    if kind == "txt":
        f, expr, fmt = args
        x = float(re.search(r"-?\d+(?:\.\d+)?", path_of(f).read_text(encoding="utf8")).group())
        out = fmtnum(repr(safe_eval(expr, {"x": x})), fmt)
        MANIFEST.append({"doc": doc, "kind": kind, "args": args, "value": out})
        return W(out)
    if kind == "cfg":
        name, fmt = args
        ns = {}
        exec((ROOT / "src" / "config.py").read_text(encoding="utf8"), {"Path": Path, "__file__": str(ROOT / "src" / "config.py")}, ns)
        out = fmtnum(str(ns[name]), fmt)
        MANIFEST.append({"doc": doc, "kind": kind, "args": args, "value": out})
        return W(out)
    if kind == "calc":
        expr, fmt = args
        out = fmtnum(repr(safe_eval(expr)), fmt)
        MANIFEST.append({"doc": doc, "kind": kind, "args": args, "value": out})
        return W(out)
    if kind == "toy":
        name, fmt = args
        out = fmtnum(repr(toy_values()[name]), fmt)
        MANIFEST.append({"doc": doc, "kind": kind, "args": args, "value": out})
        return W(out)
    if kind == "table":
        f, keycol, keys, cols, fmt = args[:5]
        names = dict(p.split("=", 1) for p in args[5].split(";")) if len(args) > 5 and args[5] else {}
        flt = args[6] if len(args) > 6 else ""
        rows = read_csv(f)
        specs = [(c.split(":", 1) + [None])[:2] for c in cols.split(";")]       # "col" or "col:fmt"
        cols = [c for c, _ in specs]
        lines = ["| " + " | ".join([keycol] + cols) + " |", "|" + "|".join(["---"] * (len(cols) + 1)) + "|"]
        for k in keys.split(";"):
            sel = (flt + "&" if flt else "") + f"{keycol}={k}"
            r = pick(rows, sel)
            if k in names:                      # key cell first: manifest order must equal the order in the text
                keycell = names[k]
            else:
                keycell = W(r[keycol])
                MANIFEST.append({"doc": doc, "kind": "csv", "args": [f, sel, keycol, "s"], "value": r[keycol]})
            cells = []
            for c, cfmt in specs:
                cf = cfmt or ("s" if c in r and not re.fullmatch(r"-?\d+(\.\d+)?(e-?\d+)?", r[c] or "x") else fmt)
                v = r[c]
                if v == "" or v.lower() == "nan":
                    cells.append("-"); continue
                out = fmtnum(v, cf)
                cells.append(W(out))
                MANIFEST.append({"doc": doc, "kind": "csv", "args": [f, sel, c, cf], "value": out})
            lines.append("| " + " | ".join([keycell] + cells) + " |")
        return "\n".join(lines)
    raise ValueError(kind)


PH = re.compile(r"\[\[(.+?)\]\]", re.S)


def render(text, doc):
    def sub(m):
        parts = [p.strip() for p in m.group(1).split("|")]
        return resolve(parts[0], parts[1:], doc)
    return PH.sub(sub, text)


# ----------------------------------------------------------------------------- real-query extracts
def make_examples():
    import numpy as np
    import pandas as pd
    import metrics as M
    from features import FEATURE_NAMES
    corpus = {}
    with open(ROOT / "data" / "raw" / "nfcorpus" / "nfcorpus" / "corpus.jsonl", encoding="utf8") as fh:
        for line in fh:
            d = json.loads(line); corpus[d["_id"]] = d["title"]
    ts = pd.read_csv(RES / "test_scores.csv.gz")
    f = pd.read_pickle(ROOT / "data" / "processed" / "full" / "features.pkl")
    f = f[f["split"] == "test"].reset_index(drop=True)
    assert (f["qid"].values == ts["qid"].values).all() and (f["doc_id"].values == ts["doc_id"].values).all()
    out = DOCS / "data_vi"; out.mkdir(exist_ok=True)
    for qid, tag in EXAMPLE_QUERIES.items():
        idx = np.flatnonzero(ts["qid"].values == qid)
        g = ts.iloc[idx].reset_index(drop=True); gf = f.iloc[idx].reset_index(drop=True)
        g["title"] = [corpus[d][:60] for d in g["doc_id"]]
        for c in ("bm25_comb", "matched_terms", "coverage_ratio", "lm_dirichlet"):
            g[c] = gf[c].values
        o = M.rank_order(g["ranksvm_sampled"].values, g["bm25_rank"].values)
        g["ltr_rank"] = 0; g.loc[o, "ltr_rank"] = np.arange(1, len(g) + 1)
        g = g.rename(columns={"ranksvm_sampled": "ltr_score"})
        b = g.sort_values("bm25_rank").head(10).copy(); b["bm25_rank"] = b["bm25_rank"].astype(int)
        b[["bm25_rank", "doc_id", "label", "bm25_comb", "matched_terms", "coverage_ratio", "ltr_rank", "title"]].to_csv(out / f"{tag}_bm25_top10.csv", index=False)
        l = g.sort_values("ltr_rank").head(10).copy(); l["bm25_rank"] = l["bm25_rank"].astype(int)
        l[["ltr_rank", "doc_id", "label", "bm25_rank", "ltr_score", "title"]].to_csv(out / f"{tag}_ltr_top10.csv", index=False)
    idx = np.flatnonzero(f["qid"].values == PAIR_QID)
    rows = []
    for role, did in PAIR_DOCS.items():
        r = f.iloc[idx][f.iloc[idx]["doc_id"] == did].iloc[0]
        lab = int(ts.iloc[idx][ts.iloc[idx]["doc_id"] == did]["label"].iloc[0])
        rows.append({"role": role, "doc_id": did, "label": lab, "title": corpus[did][:60], **{n: r[n] for n in FEATURE_NAMES}})
    pd.DataFrame(rows).to_csv(out / "example_pair_features.csv", index=False)


def main():
    make_examples()
    (DOCS / "number_manifest.json").unlink(missing_ok=True)
    for t in sorted((DOCS / "templates").glob("*.tmpl.md")):
        name = t.name.replace(".tmpl.md", ".md")
        tagged = render(t.read_text(encoding="utf8"), name)
        text = tagged.replace(L, "").replace(Rr, "")
        (DOCS / "tools" / "_tagged").mkdir(exist_ok=True)
        (DOCS / "tools" / "_tagged" / name).write_text(tagged, encoding="utf8")
        (DOCS / name).write_text(text, encoding="utf8")
        print("built", name, len(text.split()), "words")
    (DOCS / "number_manifest.json").write_text(json.dumps(MANIFEST, indent=1, ensure_ascii=False), encoding="utf8")
    print(len(MANIFEST), "numbers resolved from files")


if __name__ == "__main__":
    main()
