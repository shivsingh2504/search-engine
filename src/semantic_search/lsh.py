"""Approximate product search using random-hyperplane LSH."""

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


DATA_DIR = Path("data/processed")
EMBEDDINGS_FILE = DATA_DIR / "product_embeddings.npy"
RECORDS_FILE = DATA_DIR / "product_records.jsonl"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class LSHSearch:
    def __init__(self, bits_per_table=10, num_tables=8, seed=42):
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
        self.hyperplanes = rng.standard_normal(
            (num_tables, bits_per_table, dimension)
        ).astype(np.float32)

        self.tables = [dict() for _ in range(num_tables)]
        for row_index, vector in enumerate(self.embeddings):
            for table_index in range(num_tables):
                signature = self._signature(
                    vector, self.hyperplanes[table_index]
                )
                bucket = self.tables[table_index].setdefault(signature, [])
                bucket.append(row_index)

    @staticmethod
    def _signature(vector, hyperplanes):
        projections = hyperplanes @ vector
        bits = projections >= 0
        packed = np.packbits(bits, bitorder="little")
        return int.from_bytes(packed.tobytes(), byteorder="little")

    def _nearby_signatures(self, signature, radius=1):
        yield signature, 0

        for distance in range(1, radius + 1):
            for flipped_bits in combinations(range(self.bits_per_table), distance):
                mask = 0
                for bit_position in flipped_bits:
                    mask |= 1 << bit_position

                nearby = signature ^ mask
                hamming_distance = (signature ^ nearby).bit_count()
                yield nearby, hamming_distance

    def search(self, query, top_k=5, probe_radius=1):
        query_vector = self.model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        
        candidate_rows = set()
        for table_index in range(self.num_tables):
            query_signature = self._signature(
                query_vector, self.hyperplanes[table_index]
            )

            for nearby_signature, _distance in self._nearby_signatures(
                query_signature, radius=probe_radius
            ):
                candidate_rows.update(
                    self.tables[table_index].get(nearby_signature, [])
                )

        
        if not candidate_rows and probe_radius < self.bits_per_table:
            return self.search(query, top_k, probe_radius + 1)

        if not candidate_rows:
            return []

        candidate_rows = np.asarray(sorted(candidate_rows), dtype=np.int64)
        scores = self.embeddings[candidate_rows] @ query_vector
        order = np.argsort(scores)[::-1][:top_k]

        self.last_candidate_count = len(candidate_rows)
        return [
            {
                "score": float(scores[position]),
                **self.records[candidate_rows[position]],
            }
            for position in order
        ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Natural-language search query")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--bits", type=int, default=10)
    parser.add_argument("--tables", type=int, default=8)
    parser.add_argument("--probe-radius", type=int, default=1)
    args = parser.parse_args()

    searcher = LSHSearch(
        bits_per_table=args.bits,
        num_tables=args.tables,
    )
    results = searcher.search(
        args.query,
        top_k=args.top_k,
        probe_radius=args.probe_radius,
    )

    print(f"Candidates reranked: {getattr(searcher, 'last_candidate_count', 0)}")
    for rank, result in enumerate(results, start=1):
        print(f"\n{rank}. Score: {result['score']:.4f}")
        print(f"ID: {result['id']}")
        print(f"Text: {result['text'][:500]}")
        print(f"Metadata: {result['metadata']}")


if __name__ == "__main__":
    main()