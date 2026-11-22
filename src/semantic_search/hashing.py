import argparse
from pathlib import Path
import numpy as np

DATA_DIR = Path("data/processed")
EMBEDDINGS_FILE = DATA_DIR / "product_embeddings.npy"
SIGNATURES_FILE = DATA_DIR / "product_signatures_256.npz"

NUM_BITS = 256
BITS_PER_WORD = 64
WORDS_PER_SIG = NUM_BITS // BITS_PER_WORD
DEFAULT_SEED = 42

BIT_WEIGHTS = np.left_shift(
    np.uint64(1),
    np.arange(BITS_PER_WORD, dtype=np.uint64),
)

def make_hyperplane(input_dim,seed=DEFAULT_SEED):
  if input_dim <= 0:
    raise ValueError("input_dimension must be positive.")
  rng = np.random.default_rng(seed)
  return rng.standard_normal(NUM_BITS,input_dim).astype(np.float32)
