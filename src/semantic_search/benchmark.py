from __future__ import annotations

import argparse
import time
import statistics

import numpy as np
from .cpp_search import CppLSHSearch
from .search import LSHSearch

TOP_K = 5
QUERIES = (
    "wireless noise cancelling headphones",
    "stainless steel insulated water bottle",
    "educational building toy for kids",
    "non-slip yoga mat",
    "chef knife for the kitchen",
    "LED desk lamp for reading",
    "moisturizing face cream for dry skin",
    "phone case with card holder",
    "waterproof backpack for hiking and camping",
    "ergonomic office chair with lumbar support",
)
DEFAULT_CANDIDATES = (25, 50, 100, 250, 500)

def encode(model,query):
  return model.encode(
    query,
    normalize_embeddings=True,
    convert_to_numpy=True,
  ).astype(np.float32)
  
def percentile90(values):
  return float(np.percentile(values,90))


  