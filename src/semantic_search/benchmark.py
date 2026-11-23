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

def recall_k(reference,found,k=TOP_K):
  reference_ids = {row["id"] for row in reference[:k]}
  found_ids = {row["id"] for row in found[:k]}
  if not reference_ids:
    return 1.0
  return len(reference_ids & found_ids) / len(reference_ids)

def measure(function,warmup,repeats):
  for _ in range(warmup):
    function()
  samples = []
  for _ in range(repeats):
    start = time.perf_counter()
    function()
    samples.append(time.perf_counter() - start)
  return samples

def ms(seconds):
  return seconds * 1000


  