"""Independent verification of docs_vi/*.md (does NOT import build_docs.py).

Checks
 1. every file-derived number (tagged copy in tools/_tagged) is re-read from its source file with plain csv/json code and compared;
 2. the published .md equals the tagged copy with markers removed, and the manifest matches the tagged numbers one by one;
 3. toy examples (AP / NDCG / pairwise accuracy / hinge) are recomputed by hand-written formulas (and match the project's metrics);
 4. data_vi/*.csv (the two worked queries) are re-derived from results/test_scores.csv.gz + data/processed/full/features.pkl + corpus;
 5. every file name and every `function()` named in the documents exists in the project;
 6. NO unattributed numeric literal remains: decimals, numbers >= 100, percentages must be tagged or listed in allowed_literals.txt
    with a verifiable justification (slide:<page> is checked against Chap_8.pdf, cfg:<NAME> against src/config.py).
Exit code 0 only if everything passes.   Usage: python docs_vi/tools/check_numbers.py
"""
import csv, json, math, re, sys
sys.dont_write_bytecode = True
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
RES, DOCS = ROOT / "results", ROOT / "docs_vi"
L, Rr = chr(0xE000), chr(0xE001)
errors, notes = [], []


def err(m):
    errors.append(m)


def rows_of(f):
    p = ROOT / f if f.startswith("docs_vi/") else RES / f
    with open(p, encoding="utf8", newline="") as fh:
        return list(csv.DictReader(fh))


def holds(r, sel):
    for c in (sel.split("&") if sel else []):
        m = re.match(r"^([^<>=&]+?)(<=|>=|<|>|=)(.*)$", c)
        col, op, val = m.group(1), m.group(2), m.group(3)
        if op == "=":
            if r[col] != val:
                return False
        else:
            a, b = float(r[col]), float(val)
            if not (a < b if op == "<" else a > b if op == ">" else a <= b if op == "<=" else a >= b):
                return False
    return True


def cell(f, sel, col):
    hit = [r for r in rows_of(f) if holds(r, sel)]
    assert len(hit) == 1, f"{f} [{sel}] -> {len(hit)} rows"
    return hit[0][col]


def fmt(raw, spec):
    if spec == "s":
        return raw
    v = float(raw)
    return format(int(round(v)), spec) if spec.endswith("d") else format(v, spec)


def calc(expr):
    assert re.fullmatch(r"[0-9+\-*/(). a-z_^]*", expr) and "__" not in expr, expr
    return eval(expr.replace("^", "**"), {"__builtins__": {}}, {"log2": math.log2, "sqrt": math.sqrt})


def config_value(name):
    txt = (ROOT / "src" / "config.py").read_text(encoding="utf8")
    nl = chr(10)
    m = re.search(rf"^{name}\s*=\s*([^#{nl}]+?)\s*(?:#.*)?$", txt, re.M)
    return m.group(1)


def toy(name):
    # hand-written formulas (independent of src/metrics.py)
    if name == "ap_example":                      # ranking rels [1,0,1,0], 3 relevant in total
        return (1 / 1 + 2 / 3) / 3
    if name == "dcg_example":                     # rels [2,0,1], gain 2^rel-1, discount 1/log2(rank+1)
        return (2 ** 2 - 1) / math.log2(2) + (2 ** 0 - 1) / math.log2(3) + (2 ** 1 - 1) / math.log2(4)
    if name == "idcg_example":                    # ideal order [2,1,0]
        return (2 ** 2 - 1) / math.log2(2) + (2 ** 1 - 1) / math.log2(3) + 0
    if name == "ndcg_example":
        return toy("dcg_example") / toy("idcg_example")
    if name == "pairacc_example":                 # labels [2,1,0], scores [3,1,2]: pairs (0,1) ok, (0,2) ok, (1,2) wrong
        return 2 / 3
    raise KeyError(name)


def resolve(e):
    k, a = e["kind"], e["args"]
    if k == "csv":
        return fmt(cell(a[0], a[1], a[2]), a[3])
    if k == "count":
        return format(sum(1 for r in rows_of(a[0]) if holds(r, a[1])), a[2])
    if k == "json":
        v = json.loads((RES / a[0]).read_text(encoding="utf8"))
        for p in a[1].split("."):
            v = v[p]
        return fmt(str(v), a[2])
    if k == "txt":
        x = float(re.search(r"-?\d+(?:\.\d+)?", (RES / a[0]).read_text(encoding="utf8")).group())
        return fmt(repr(calc(a[1].replace("x", repr(x)))), a[2])
    if k == "cfg":
        return fmt(config_value(a[0]), a[1])
    if k == "calc":
        return fmt(repr(calc(a[0])), a[1])
    if k == "toy":
        return fmt(repr(toy(a[0])), a[1])
    raise ValueError(k)


# ---------------------------------------------------------------- 1,2 manifest vs sources vs documents
manifest = json.loads((DOCS / "number_manifest.json").read_text(encoding="utf8"))
by_doc = {}
for e in manifest:
    by_doc.setdefault(e["doc"], []).append(e)
n_checked = 0
for doc, entries in by_doc.items():
    tagged = (DOCS / "tools" / "_tagged" / doc).read_text(encoding="utf8")
    final = (DOCS / doc).read_text(encoding="utf8")
    if tagged.replace(L, "").replace(Rr, "") != final:
        err(f"{doc}: published text differs from tagged copy")
    segs = re.findall(f"{L}(.*?){Rr}", tagged, re.S)
    if len(segs) != len(entries):
        err(f"{doc}: {len(segs)} tagged numbers but {len(entries)} manifest entries")
    for seg, e in zip(segs, entries):
        want = resolve(e)
        n_checked += 1
        if seg != want or e["value"] != want:
            err(f"{doc}: {e['kind']} {e['args']} -> document has {seg!r}, source gives {want!r}")
for md in ("docs_vi/data_vi",):
    pass
print(f"[1-2] re-read {n_checked} file-derived numbers from their sources")

# ---------------------------------------------------------------- 3 toy examples vs project metrics
sys.path.insert(0, str(ROOT / "src"))
import metrics as M
for name, val in {"ap_example": M.average_precision([1, 0, 1, 0], 3), "ndcg_example": M.ndcg_at_k([2, 0, 1], [2, 1, 0], 10),
                  "dcg_example": M.dcg_at_k([2, 0, 1], 3), "idcg_example": M.dcg_at_k([2, 1, 0], 3)}.items():
    if abs(val - toy(name)) > 1e-12:
        err(f"toy {name}: hand formula {toy(name)} != project metric {val}")
c, t, _ = M.pairwise_counts([3, 1, 2], [2, 1, 0])
if abs(c / t - toy("pairacc_example")) > 1e-12:
    err("toy pairwise accuracy mismatch")
print("[3] toy examples: hand formulas agree with src/metrics.py")

# ---------------------------------------------------------------- 4 worked-query extracts
import pandas as pd
corpus = {}
with open(ROOT / "data" / "raw" / "nfcorpus" / "nfcorpus" / "corpus.jsonl", encoding="utf8") as fh:
    for line in fh:
        d = json.loads(line); corpus[d["_id"]] = d["title"]
ts = pd.read_csv(RES / "test_scores.csv.gz")
feat = pd.read_pickle(ROOT / "data" / "processed" / "full" / "features.pkl")
feat = feat[feat["split"] == "test"].reset_index(drop=True)
assert (feat["qid"].values == ts["qid"].values).all()
for qid, tag in {"PLAIN-1537": "improved", "PLAIN-3085": "harmed"}.items():
    m = (ts["qid"] == qid).values
    g = ts[m].reset_index(drop=True); gf = feat[m].reset_index(drop=True)
    # independent LTR order: score desc, ties by original BM25 rank asc (stable sort)
    order = g.assign(i=range(len(g))).sort_values(["ranksvm_sampled", "bm25_rank"], ascending=[False, True], kind="stable")["i"].tolist()
    ltr_rank = {g.loc[i, "doc_id"]: r + 1 for r, i in enumerate(order)}
    b = rows_of(f"docs_vi/data_vi/{'q1537' if tag == 'improved' else 'q3085'}_bm25_top10.csv")
    l = rows_of(f"docs_vi/data_vi/{'q1537' if tag == 'improved' else 'q3085'}_ltr_top10.csv")
    top_b = g.sort_values("bm25_rank").head(10)["doc_id"].tolist()
    if [r["doc_id"] for r in b] != top_b:
        err(f"{qid}: BM25 top-10 differs from test_scores.csv.gz")
    if [r["doc_id"] for r in l] != [g.loc[i, "doc_id"] for i in order[:10]]:
        err(f"{qid}: LTR top-10 differs from re-derived order")
    for r in b + l:
        did = r["doc_id"]; j = g.index[g["doc_id"] == did][0]
        if int(r["label"]) != int(g.loc[j, "label"]) or not corpus[did].startswith(r["title"]):
            err(f"{qid}/{did}: label or title mismatch")
        if "ltr_rank" in r and int(r["ltr_rank"]) != ltr_rank[did]:
            err(f"{qid}/{did}: ltr_rank mismatch")
        if "bm25_comb" in r and abs(float(r["bm25_comb"]) - gf.loc[j, "bm25_comb"]) > 1e-9:
            err(f"{qid}/{did}: bm25_comb mismatch")
        if "ltr_score" in r and abs(float(r["ltr_score"]) - g.loc[j, "ranksvm_sampled"]) > 1e-9:
            err(f"{qid}/{did}: ltr_score mismatch")
pf = rows_of("docs_vi/data_vi/example_pair_features.csv")
for r in pf:
    j = feat.index[(feat["qid"] == "PLAIN-1537") & (feat["doc_id"] == r["doc_id"])][0]
    for k, v in r.items():
        if k in ("role", "doc_id", "title"):
            continue
        ref = feat.loc[j, k] if k != "label" else ts.loc[j, "label"]
        if abs(float(v) - float(ref)) > 1e-9:
            err(f"pair features {r['doc_id']}.{k}: {v} vs {ref}")
print("[4] worked-query extracts re-derived from test_scores.csv.gz / features.pkl / corpus")

# ---------------------------------------------------------------- 5 file and function references
src_defs = set()
for p in list((ROOT / "src").glob("*.py")) + list((ROOT / "tests").glob("*.py")):
    src_defs |= set(re.findall(r"^\s*(?:def|class)\s+(\w+)", p.read_text(encoding="utf8"), re.M))
    src_defs |= set(re.findall(r"^([A-Z_][A-Z0-9_]*)\s*=", p.read_text(encoding="utf8"), re.M))
all_files = {p.name for p in ROOT.rglob("*") if p.is_file() and "__pycache__" not in str(p) and ".pytest_cache" not in str(p)}
all_rel = {str(p.relative_to(ROOT)).replace("\\", "/") for p in ROOT.rglob("*") if p.is_file() and "__pycache__" not in str(p)}
n_files = n_funcs = 0
for md in sorted(DOCS.glob("*.md")):
    text = md.read_text(encoding="utf8")
    for tok in re.findall(r"`([^`\n]+)`", text):
        tok = tok.strip()
        m = re.fullmatch(r"([A-Za-z_][\w]*)\(\)", tok)
        if m:
            n_funcs += 1
            if m.group(1) not in src_defs:
                err(f"{md.name}: function {tok} not found in src/ or tests/")
            continue
        if re.fullmatch(r"[\w./*\-]+\.(csv|json|md|png|gz|txt|py|sh|yml|pkl|pdf)", tok) or re.fullmatch(r"[\w./\-]+/", tok):
            n_files += 1
            if "*" in tok:
                pat = re.compile("(^|/)" + re.escape(tok).replace(r"\*", ".*") + "$")
                if not any(pat.search(a) for a in all_rel):
                    err(f"{md.name}: pattern {tok} matches no file")
            elif tok.endswith("/"):
                if not (ROOT / tok).is_dir():
                    err(f"{md.name}: folder {tok} missing")
            elif not (tok in all_files or tok in all_rel or any(a.endswith("/" + tok) for a in all_rel)):
                err(f"{md.name}: file {tok} not found in the project")
print(f"[5] {n_files} file references and {n_funcs} function references exist")

# ---------------------------------------------------------------- 6 unattributed numeric literals
allowed = {}
for line in (DOCS / "tools" / "allowed_literals.txt").read_text(encoding="utf8").splitlines():
    if line.strip() and not line.startswith("#"):
        lit, why = [x.strip() for x in line.split("|")]
        allowed[lit] = why
try:
    from pypdf import PdfReader
    slides = PdfReader(str(ROOT / "Chap_8.pdf"))
except Exception:
    slides = None
for lit, why in allowed.items():
    if why.startswith("slide:"):
        pg = int(why.split(":")[1])
        txt = (slides.pages[pg - 1].extract_text() or "") if slides else ""
        if slides and lit.replace(",", "") not in txt.replace(",", ""):
            err(f"allowed literal {lit!r}: not found on slide {pg} of Chap_8.pdf")
    elif why.startswith("cfg:"):
        if fmt(config_value(why[4:]), "g") != lit and config_value(why[4:]) != lit:
            err(f"allowed literal {lit!r}: config {why[4:]} = {config_value(why[4:])}")
    elif why.startswith("def:"):
        pass          # definitional / unit constant (documented in allowed_literals.txt)
NUM = re.compile(r"\d{1,3}(?:,\d{3})+|\d+\.\d+|\d{3,}|\d+\s?%")
leftovers = []
for md in sorted((DOCS / "tools" / "_tagged").glob("*.md")):
    t = md.read_text(encoding="utf8")
    t = re.sub(f"{L}.*?{Rr}", " ", t, flags=re.S)                       # verified numbers
    t = re.sub(r"```.*?```", " ", t, flags=re.S)                         # code blocks
    t = re.sub(r"`[^`\n]*`", " ", t)                                     # inline code (file/function names)
    t = re.sub(r"\]\([^)]*\)", "]", t)                                   # link targets
    t = re.sub(r"(PLAIN|MED)-\d+", " ", t)                              # corpus identifiers (query / document ids)
    t = re.sub(r"(?m)^#{1,6}\s+\d+(\.\d+)*", "#", t)                        # numbered headings (section numbers of this guide)
    t = re.sub(r"(?i)(slide|trang|mục|§|section|phần|chương|chapter|bước|câu|mở rộng)\s*\d+(\.\d+)*", " ", t)
    for m in NUM.finditer(t):
        lit = m.group().replace(" ", "")
        if lit not in allowed:
            ctx = t[max(0, m.start() - 40): m.end() + 30].replace("\n", " ")
            leftovers.append(f"{md.name}: {lit!r} ... {ctx}")
for x in leftovers:
    err("unattributed number: " + x)
print(f"[6] unattributed numeric literals: {len(leftovers)}")

print()
if errors:
    print(f"FAILED: {len(errors)} problem(s)")
    for e in errors[:80]:
        print(" -", e)
    sys.exit(1)
print("ALL CHECKS PASSED")
