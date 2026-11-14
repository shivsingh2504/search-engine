# Speed Semantic Search with LSH

A semantic product-search project. It will embed product text, retrieve candidates with random-hyperplane locality-sensitive hashing, and compare that approximate search with exact cosine search.

## Phase 1: prepare data

The sample dataset is stored locally at `data/raw/amazon_products.parquet` and is intentionally excluded from Git. To normalize product text into one JSONL record per product, run from the repository root:

```powershell
.\.venv\Scripts\Activate.ps1
python -m src.semantic_search.data data\raw\amazon_products.parquet --text-columns title description features categories --id-column parent_asin --exclude-metadata images --output data\processed\products.jsonl
```

Each output line contains an `id`, a cleaned `text` field, and `metadata`. The source and processed datasets are excluded from Git so the repository contains code rather than copied data.

The loader supports CSV, JSONL, and Parquet input. Choose text columns that describe the records; other columns are retained as metadata unless named in `--exclude-metadata`.
