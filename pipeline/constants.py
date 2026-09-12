"""Paths, seeds and parameters shared by every stage of the pipeline."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = ROOT / "data/raw/ft911"
STOPWORDS_PATH = ROOT / "data/stopwords.txt"
TOPICS_PATH = ROOT / "data/trec/topics.301-450.601-700.txt"
QRELS_PATH = ROOT / "data/trec/qrels.ft911.txt"
OUT_DIR = ROOT / "site/public/data"
GOLDEN_DIR = ROOT / "tests/golden"
STOPWORDS_JSON = ROOT / "site/src/search/stopwords.json"

SEED = 20260912
N_BOOT = 10_000

BM25_K1 = 1.2
BM25_B = 0.75
GRID_K1 = [0.6, 0.9, 1.2, 1.5, 2.0]
GRID_B = [0.0, 0.25, 0.5, 0.75, 1.0]

SIZE_LIMIT = 200_000
TOP_N = 1000

# treatment key -> (stem, remove stopwords)
TREATMENTS = {
    "stem_stop": (True, True),
    "nostem_stop": (False, True),
    "stem_nostop": (True, False),
    "nostem_nostop": (False, False),
}
TREATMENT_LABELS = {
    "stem_stop": "Stemmed, stopwords removed",
    "nostem_stop": "Unstemmed, stopwords removed",
    "stem_nostop": "Stemmed, stopwords kept",
    "nostem_nostop": "Unstemmed, stopwords kept",
}
