import argparse
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

from itertools import combinations

DATA_DIR = Path("data/processed")
EMBEDDINGS_FILE = DATA_DIR / "product_embeddings.npy"
RECORDS_FILE = DATA_DIR / "product_records.jsonl"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"