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
  
  def search(self,query,top_k=5):
    query_vector = self.model.encode(
      query,
      normalize_embeddings=True,
      convert_to_numpy=True,
    ).astype(np.float32)
    return self.search_vector(query_vector, top_k=top_k)

  def search_vector(self, query_vector, top_k=5):
    """Search with an already embedded, unit-normalized query vector."""
    query_vector = np.asarray(query_vector, dtype=np.float32)
    if query_vector.ndim != 1 or query_vector.shape[0] != self.embeddings.shape[1]:
      raise ValueError(
        f"query_vector must have shape ({self.embeddings.shape[1]},); "
        f"got {query_vector.shape}."
      )
    scores = self.embeddings @ query_vector
    top_k = min(top_k, len(self.records))
    best_indices = np.argsort(scores)[::-1][:top_k]
    return [
      {
        "score": float(scores[index]),
        **self.records[index],
      }
      for index in best_indices
    ]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Natural-language product search query")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    searcher = ExactSearch()
    results = searcher.search(args.query, args.top_k)

    for rank, result in enumerate(results, start=1):
        print(f"\n{rank}. Score: {result['score']:.4f}")
        print(f"ID: {result['id']}")
        print(f"Text: {result['text'][:500]}")
        print(f"Metadata: {result['metadata']}")


if __name__ == "__main__":
    main()
