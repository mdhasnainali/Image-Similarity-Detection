"""
STEP 2 - Find the Most Similar Image Pairs
===========================================
What this script does:
  - Loads the feature vectors saved by Step 1
  - Reduces their size with PCA (makes the search faster)
  - Builds a FAISS index (a very fast search structure)
  - For every image, finds its single closest neighbour
  - Keeps only the top N most similar pairs (no duplicates)
  - Saves the results to a CSV file you can open in Excel / Google Sheets

How to run:
  python step2_find_similar_pairs.py

Output files:
  pickle/pca_features.pickle   - PCA-compressed features (intermediate)
  pickle/faiss_index.pickle    - serialised FAISS index (intermediate)
  output/similar_pairs.csv     - the final results table
"""

import os
import pickle
import csv

import numpy as np
import faiss
from sklearn.decomposition import PCA
from tqdm import tqdm

# ── Configuration ────────────────────────────────────────────────────────────
PICKLE_DIR = "pickle"
OUTPUT_DIR = "output"

FEATURES_FILE  = os.path.join(PICKLE_DIR, "features.pickle")
FILENAMES_FILE = os.path.join(PICKLE_DIR, "filenames.pickle")
PCA_FILE       = os.path.join(PICKLE_DIR, "pca_features.pickle")
FAISS_FILE     = os.path.join(PICKLE_DIR, "faiss_index.pickle")
CSV_FILE       = os.path.join(OUTPUT_DIR, "similar_pairs.csv")

PCA_DIMENSIONS = 200   # reduce 2048-d vectors to 200-d (keeps 95 %+ of info)
TOP_N_PAIRS    = 100   # how many most-similar pairs to write to the CSV
# ─────────────────────────────────────────────────────────────────────────────


def load_pickles():
    """Load features and filenames saved by Step 1."""
    with open(FEATURES_FILE, "rb") as f:
        features = pickle.load(f)
    with open(FILENAMES_FILE, "rb") as f:
        filenames = pickle.load(f)
    return features, filenames


def compress_with_pca(features, n_components):
    """Fit PCA on the feature matrix and return the compressed version."""
    matrix = np.array(features, dtype=np.float32)   # shape: [N, 2048]
    n_components = min(n_components, matrix.shape[0], matrix.shape[1])
    pca = PCA(n_components=n_components)
    compressed = pca.fit_transform(matrix).astype(np.float32)
    return compressed


def build_faiss_index(compressed):
    """Build a flat L2 FAISS index from the compressed feature matrix."""
    dim = compressed.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(compressed)
    return index


def find_similar_pairs(index, compressed, filenames, top_n):
    """
    For each image query its nearest neighbour (excluding itself).
    Collect all (image_a, image_b, distance) triples, deduplicate, sort by
    distance, and return the top_n closest pairs.
    """
    # Search for 2 neighbours: the image itself (distance=0) + the closest other
    distances, indices = index.search(compressed, 2)

    seen = set()
    pairs = []

    for i in tqdm(range(len(filenames)), desc="Finding pairs", unit="img"):
        neighbour_idx  = indices[i][1]   # index 0 is the image itself
        neighbour_dist = float(distances[i][1])

        # Build a canonical key so (A,B) and (B,A) are treated as the same pair
        key = tuple(sorted((i, neighbour_idx)))
        if key in seen:
            continue
        seen.add(key)

        pairs.append({
            "image_a":  filenames[i],
            "image_b":  filenames[neighbour_idx],
            "distance": round(neighbour_dist, 6),
            # Similarity score: 1 means identical, 0 means very different
            "similarity_score": round(max(0.0, 1.0 - neighbour_dist / 2.0), 6),
        })

    # Sort by distance ascending (most similar first)
    pairs.sort(key=lambda x: x["distance"])
    return pairs[:top_n]


def save_csv(pairs, csv_path):
    """Write the pairs list to a CSV file."""
    fieldnames = ["rank", "image_a", "image_b", "distance", "similarity_score"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, pair in enumerate(pairs, start=1):
            writer.writerow({"rank": rank, **pair})


def main():
    # ── Sanity checks ────────────────────────────────────────────────────────
    for path in (FEATURES_FILE, FILENAMES_FILE):
        if not os.path.exists(path):
            print(f"[ERROR] Required file not found: '{path}'")
            print("  Please run  python step1_extract_features.py  first.")
            return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────────────
    print("[1/5] Loading features from pickle/ …")
    features, filenames = load_pickles()
    print(f"      Loaded {len(filenames)} images.")

    if len(filenames) < 2:
        print("[ERROR] Need at least 2 images to find similar pairs.")
        return

    # ── PCA compression ──────────────────────────────────────────────────────
    print(f"[2/5] Compressing features with PCA ({PCA_DIMENSIONS} dimensions) …")
    compressed = compress_with_pca(features, PCA_DIMENSIONS)
    with open(PCA_FILE, "wb") as f:
        pickle.dump(compressed, f)
    print(f"      Saved → {PCA_FILE}")

    # ── Build FAISS index ────────────────────────────────────────────────────
    print("[3/5] Building FAISS search index …")
    index = build_faiss_index(compressed)
    # Serialise the index via faiss's own serialiser then store in pickle
    index_bytes = faiss.serialize_index(index).tobytes()
    with open(FAISS_FILE, "wb") as f:
        pickle.dump(index_bytes, f)
    print(f"      Saved → {FAISS_FILE}")

    # ── Find similar pairs ───────────────────────────────────────────────────
    print(f"[4/5] Finding the {TOP_N_PAIRS} most similar pairs …")
    pairs = find_similar_pairs(index, compressed, filenames, TOP_N_PAIRS)

    # ── Save CSV ─────────────────────────────────────────────────────────────
    print("[5/5] Saving results to CSV …")
    save_csv(pairs, CSV_FILE)
    print(f"      Saved → {CSV_FILE}")

    print(f"\n✅ Done!  Top {len(pairs)} most similar pairs written to:")
    print(f"   {CSV_FILE}")
    print("\nNext step: run  python step3_visualize.py")


if __name__ == "__main__":
    main()
