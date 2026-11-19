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

class LSHSearch:
  def __init__(self,bits_per_table=10 , num_tables=8, seed=42):
    self.embeddings = np.load(EMBEDDINGS_FILE).astype(np.float32)
    with RECORDS_FILE.open("r", encoding="utf-8") as file:
            self.records = [json.loads(line) for line in file if line.strip()]

    if len(self.embeddings) != len(self.records):
      raise ValueError("Embedding and record counts do not match.")
    
    self.model = SentenceTransformer(MODEL_NAME)
    self.bits_per_table = bits_per_table
    self.num_tables = num_tables
    
    rng = np.random.default_rng(seed)
    dimension = self.embeddings.shape[1]
    self.hyper = rng.standard_normal(
      (num_tables,bits_per_table,dimension)
    ).astype(np.float32)
    self.tables = [dict() for _ in range(num_tables)]
    for row_index, vector in enumerate(self.embeddings):
      for table_index in range(num_tables):
        signature = self._signature(
        vector, self.hyperplanes[table_index]
        )
        bucket = self.tables[table_index].setdefault(signature, [])
        bucket.append(row_index)
