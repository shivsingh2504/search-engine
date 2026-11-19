import argparse
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

DATA_DIR = Path("data/processed")
EMBEDDINGS_FILE = DATA_DIR / "product_embeddings.npy"
RECORDS_FILE = DATA_DIR / "product_records.jsonl"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

