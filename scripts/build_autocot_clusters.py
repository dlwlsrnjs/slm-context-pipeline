#!/usr/bin/env python3
"""
Auto-CoT style clustering for math_5k seed pools.

For each dataset (gsm8k_5k, orcamath_5k, metamath_5k):
  1. Encode questions with all-MiniLM-L6-v2
  2. KMeans cluster (k=8 by default)
  3. For each cluster, pick the sample CLOSEST to the centroid as representative
  4. Save: cluster centroid representatives (8 per dataset) + per-sample cluster assignment

The representatives become a high-quality, diverse seed pool for RL.
Instead of randomly picking 4 of 5000, we pick 4 of 8 cluster reps per prompt.

Usage:
    python scripts/build_autocot_clusters.py --num-clusters 8
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

REPO_ROOT = Path(__file__).resolve().parent.parent
SEED_POOL_PATHS = {
    "gsm8k_5k": REPO_ROOT / "slm_context_pipeline/data/math_5k/gsm8k_5k.jsonl",
    "orcamath_5k": REPO_ROOT / "slm_context_pipeline/data/math_5k/orcamath_5k.jsonl",
    "metamath_5k": REPO_ROOT / "slm_context_pipeline/data/math_5k/metamath_5k.jsonl",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num-clusters", type=int, default=8,
                   help="k for KMeans (Auto-CoT default = 8 for gsm8k)")
    p.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--seed", type=int, default=192,
                   help="Auto-CoT default random seed")
    p.add_argument("--output-dir",
                   default="slm_context_pipeline/data/math_5k_clusters")
    return p.parse_args()


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def cluster_dataset(rows, encoder, k, seed):
    questions = [r["question"] for r in rows]
    print(f"  encoding {len(questions)} questions...")
    embeddings = encoder.encode(questions, show_progress_bar=False, batch_size=128)

    print(f"  k-means k={k}...")
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    km.fit(embeddings)
    labels = km.labels_

    # distance(sample, all centroids) shape (N, k); we want distance to OWN centroid
    dist_to_all = km.transform(embeddings)
    own_dist = dist_to_all[np.arange(len(rows)), labels]

    # for each cluster, find sample with min distance to centroid
    representatives = []
    for c in range(k):
        idx_in_cluster = np.where(labels == c)[0]
        best_local = idx_in_cluster[np.argmin(own_dist[idx_in_cluster])]
        rep = rows[best_local]
        representatives.append({
            "cluster_id": int(c),
            "rep_index": int(best_local),
            "rep_id": rep["id"],
            "question": rep["question"],
            "answer": rep["answer"],
            "solution": rep.get("solution", ""),
            "cluster_size": int((labels == c).sum()),
            "distance_to_centroid": float(own_dist[best_local]),
        })

    # per-sample assignments (id → cluster_id)
    assignments = {row["id"]: int(label) for row, label in zip(rows, labels)}

    return representatives, assignments


def main():
    args = parse_args()
    out_dir = REPO_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading encoder: {args.encoder}")
    encoder = SentenceTransformer(args.encoder)

    summary = {}
    for name, path in SEED_POOL_PATHS.items():
        print(f"\n=== {name} ===")
        rows = load_jsonl(path)
        reps, assignments = cluster_dataset(rows, encoder, args.num_clusters, args.seed)

        out_file = out_dir / f"{name}_clusters_k{args.num_clusters}.json"
        out_file.write_text(json.dumps({
            "dataset": name,
            "num_clusters": args.num_clusters,
            "encoder": args.encoder,
            "seed": args.seed,
            "representatives": reps,
            "assignments": assignments,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"  saved {out_file}")
        cluster_sizes = [r["cluster_size"] for r in reps]
        print(f"  cluster sizes: min={min(cluster_sizes)}, "
              f"max={max(cluster_sizes)}, mean={np.mean(cluster_sizes):.0f}")
        for r in reps[:3]:
            print(f"  cluster {r['cluster_id']} (size {r['cluster_size']}): "
                  f"{r['question'][:80]}...")
        summary[name] = {
            "n_samples": len(rows),
            "num_clusters": args.num_clusters,
            "cluster_sizes": cluster_sizes,
        }

    summary_file = out_dir / "summary.json"
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[DONE] summary → {summary_file}")


if __name__ == "__main__":
    main()
