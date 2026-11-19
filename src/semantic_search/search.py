import argparse
from pathlib import Path
import json
import numpy as np
from sentence_transformers import SentenceTransformer

DATA_DIR = Path("data/processed")
EMBEDDINGS_FILE = DATA_DIR / "product_embeddings.npy"
RECORDS_FILE = DATA_DIR / "product_records.jsonl"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

class ExactSearch:
  def __init__(self):
    self.embeddings = np.load(EMBEDDINGS_FILE)
    with RECORDS_FILE.open("r",encoding="utf-8") as file:
      self.records = [json.loads(line) for line in file if line.strip()]
    if len(self.embeddings)!= len(self.records):
      raise ValueError(
        "The number of embeddings does not match the number of records."
      )
    self.model = SentenceTransformer(MODEL_NAME)
    

    