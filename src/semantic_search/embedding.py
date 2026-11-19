import json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

INPUT_FILE = Path("data/processed/products.jsonl")
OUTPUT_DIR = Path("data/processed")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def main():
  with INPUT_FILE.open("r",encoding="utf-8") as file:
    records = [json.loads(line) for line in file if line.strip()]
    
  texts = [record["text"] for record in records]
  if not texts:
    raise ValueError(f"No product texts found in {INPUT_FILE}")
  
  model = SentenceTransformer(MODEL_NAME)
  embeddings = model.encode(
    texts,
    batch_size = 32,
    show_progress_bar = True,
    normalize_embeddings = True,
    convert_to_numpy = True,
  ).astype(np.float32)
  OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
  np.save(OUTPUT_DIR / "embeddings.npy", embeddings)
  with (OUTPUT_DIR / "product_records.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
  print(f"Embedded {len(texts):,} products")
  print(f"Vector shape: {embeddings.shape}")
  print(f"Saved vectors to {OUTPUT_DIR / 'product_embeddings.npy'}")


if __name__ == "__main__":
    main()