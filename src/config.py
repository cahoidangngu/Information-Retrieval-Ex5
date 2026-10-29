"""Central configuration: paths, seeds, hyper-parameters."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
DATA_DIR = RAW_DIR / "nfcorpus" / "nfcorpus"      # corpus.jsonl, queries.jsonl, qrels/
PROC_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
DATA_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip"

SEED = 42
TOP_K = 100                      # BM25 candidates per query
BM25_K1, BM25_B = 1.2, 0.75      # defaults, NOT tuned
LM_MU = 1000.0                   # Dirichlet prior, fixed (not tuned)
C_GRID = [0.001, 0.01, 0.1, 1, 10, 100]
SELECT_METRIC = "NDCG@10"        # validation metric used to choose C / best LTR model

# Ranking SVM pair construction
PAIR_MEM_BUDGET_GB = 2.0         # if estimated all-pairs training memory exceeds this -> sample
PAIR_CAP_PER_QUERY = 200         # sampled variant: uniform sample of at most this many valid pairs / query

# Significance
N_BOOT = 10000
