import argparse
from inspect import signature
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
  return rng.standard_normal((NUM_BITS,input_dim)).astype(np.float32)

def pack_sig(embeddings,hyperplanes,batch_size=8192):
  embeddings = np.asarray(embeddings,dtype=np.float32)
  hyperplanes = np.asarray(hyperplanes,dtype=np.float32)
  if embeddings.ndim != 2:
    raise ValueError(
      f"embeddings must have shape (N, D); got {embeddings.shape}."
    )
  if hyperplanes.shape != (NUM_BITS,embeddings.shape[1]):
    raise ValueError(
      f"hyperplanes must have shape ({NUM_BITS}, {embeddings.shape[1]}); "
      f"got {hyperplanes.shape}."
    )
  if batch_size <= 0:
    raise ValueError("batch_size must be positive.")
  if not np.isfinite(embeddings).all():
      raise ValueError("embeddings contain NaN or infinity.")
  if not np.isfinite(hyperplanes).all():
      raise ValueError("hyperplanes contain NaN or infinity.")
  sig = np.empty(
    (len(embeddings),WORDS_PER_SIG),
    dtype=np.uint64
  )
  
  for start in range(0,len(embeddings),batch_size):
    block = embeddings[start: start+batch_size]
    bits = (block @ hyperplanes.T) >= 0
    bits = bits.reshape(
      len(block),WORDS_PER_SIG,BITS_PER_WORD).astype(np.uint64)
    sig[start : start+len(block)] = np.sum(
      bits*BIT_WEIGHTS,
      axis=2,
      dtype = np.uint64
    )
  return np.ascontiguousarray(sig)

def hash_query(vector, hyperplanes):
    vector = np.asarray(vector, dtype=np.float32)

    if vector.ndim != 1:
        raise ValueError(f"query vector must be one-dimensional; got {vector.shape}.")

    signatures = pack_sig(
        vector.reshape(1, -1),
        hyperplanes,
        batch_size=1,
    )
    return signatures[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--batch-size", type=int, default=8192)
    args = parser.parse_args()

    if not EMBEDDINGS_FILE.is_file():
        raise FileNotFoundError(
            f"Embeddings not found at {EMBEDDINGS_FILE}. "
            "Run the embedding step first."
        )

    embeddings = np.load(EMBEDDINGS_FILE)
    hyperplanes = make_hyperplane(embeddings.shape[1], seed=args.seed)
    signatures = pack_sig(
        embeddings,
        hyperplanes,
        batch_size=args.batch_size,
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        SIGNATURES_FILE,
        signatures=signatures,
        hyperplanes=hyperplanes,
        seed=np.int64(args.seed),
    )

    print(f"Products hashed: {len(signatures):,}")
    print(f"Signature shape: {signatures.shape}")
    print(f"Signature dtype: {signatures.dtype}")
    print(f"Saved signatures and hyperplanes to: {SIGNATURES_FILE}")

if __name__ == "__main__":
    main()