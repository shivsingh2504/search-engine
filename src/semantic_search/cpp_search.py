
import argparse

import numpy as np
from sentence_transformers import SentenceTransformer

from ._lsh_cpp import HammingIndex
from .data import load_records
from .hashing import (
    EMBEDDINGS_FILE,
    SIGNATURES_FILE,
    hash_query,
)
from .search import MODEL_NAME, RECORDS_FILE


class CppLSHSearch:
    def __init__(self):
        self.embeddings = np.load(EMBEDDINGS_FILE).astype(np.float32)
        self.records = load_records(RECORDS_FILE)

        with np.load(SIGNATURES_FILE) as saved:
            signatures = np.ascontiguousarray(saved["signatures"], dtype=np.uint64)
            self.hyperplanes = saved["hyperplanes"].copy()

        if not len(self.embeddings) == len(self.records) == len(signatures):
            raise ValueError(
                "Embeddings, records, and signatures have different row counts."
            )
        if self.hyperplanes.shape != (
            signatures.shape[1] * 64,
            self.embeddings.shape[1],
        ):
            raise ValueError("Hyperplane shape does not match the embeddings.")

        self.index = HammingIndex(signatures)
        self.model = SentenceTransformer(MODEL_NAME)

    def search(self, query, top_k=5, candidates=100, rerank=True):
        if top_k < 1 or candidates < 1:
            raise ValueError("top_k and candidates must be at least 1.")

        query_vector = self.model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        # C++ finds the nearest binary signatures by Hamming distance.
        search_k = max(top_k, candidates) if rerank else top_k
        query_signature = hash_query(query_vector, self.hyperplanes)
        rows, distances = self.index.search(query_signature, search_k)

        # Optionally rerank those rows with their original float embeddings.
        if rerank:
            scores = self.embeddings[rows] @ query_vector
            order = np.argsort(-scores, kind="stable")[:top_k]
        else:
            scores = None
            order = np.arange(min(top_k, len(rows)))

        return [
            {
                "score": float(scores[position]) if rerank else None,
                "hamming_distance": int(distances[position]),
                **self.records[int(rows[position])],
            }
            for position in order
        ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Natural-language product query")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidates", type=int, default=100)
    parser.add_argument("--no-rerank", action="store_true")
    args = parser.parse_args()

    searcher = CppLSHSearch()
    results = searcher.search(
        args.query,
        top_k=args.top_k,
        candidates=args.candidates,
        rerank=not args.no_rerank,
    )

    for rank, result in enumerate(results, start=1):
        score = "not reranked" if result["score"] is None else f"{result['score']:.4f}"
        print(
            f"\n{rank}. Cosine score: {score}; "
            f"Hamming distance: {result['hamming_distance']}"
        )
        print(f"ID: {result['id']}")
        print(f"Text: {result['text'][:500]}")
        print(f"Metadata: {result['metadata']}")


if __name__ == "__main__":
    main()