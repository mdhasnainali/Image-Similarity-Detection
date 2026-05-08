"""
STEP 3 - Visualize Similar Pairs
==================================
What this script does:
  - Reads the CSV produced by Step 2
  - For each of the top pairs, saves a side-by-side comparison image
    (image A on the left, image B on the right, similarity score in the title)
  - Also creates a t-SNE scatter plot so you can see how all images cluster
    in feature space (images that look alike end up close together)

How to run:
  python step3_visualize.py

Output files (saved in output/):
  pairs/pair_001.png … pair_NNN.png  - side-by-side comparison images
  tsne_plot.png                      - 2-D scatter plot of all images
"""

import os
import csv
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")          # no display needed – saves files directly
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.manifold import TSNE
from tqdm import tqdm

# ── Configuration ────────────────────────────────────────────────────────────
PICKLE_DIR  = "pickle"
OUTPUT_DIR  = "output"
PAIRS_DIR   = os.path.join(OUTPUT_DIR, "pairs")

CSV_FILE    = os.path.join(OUTPUT_DIR, "similar_pairs.csv")
PCA_FILE    = os.path.join(PICKLE_DIR, "pca_features.pickle")
FILENAMES_FILE = os.path.join(PICKLE_DIR, "filenames.pickle")
TSNE_PLOT   = os.path.join(OUTPUT_DIR, "tsne_plot.png")

MAX_PAIRS_TO_VISUALIZE = 20   # how many side-by-side images to create
TSNE_MAX_IMAGES        = 500  # t-SNE is slow; cap at this many images
# ─────────────────────────────────────────────────────────────────────────────


def load_csv(csv_path):
    """Return a list of dicts, one per row in the CSV."""
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def save_pair_image(path_a, path_b, similarity, out_path):
    """Create and save a side-by-side comparison of two images."""
    try:
        img_a = Image.open(path_a).convert("RGB").resize((300, 300))
        img_b = Image.open(path_b).convert("RGB").resize((300, 300))
    except (FileNotFoundError, OSError):
        return  # skip if an image is missing

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(img_a)
    axes[0].set_title(os.path.basename(path_a), fontsize=9)
    axes[0].axis("off")

    axes[1].imshow(img_b)
    axes[1].set_title(os.path.basename(path_b), fontsize=9)
    axes[1].axis("off")

    fig.suptitle(f"Similarity Score: {float(similarity):.4f}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def save_tsne_plot(compressed, filenames, out_path, max_images):
    """Run t-SNE on (up to max_images) feature vectors and save a scatter plot."""
    n = min(len(filenames), max_images)
    subset = compressed[:n].astype(np.float32)

    print(f"      Running t-SNE on {n} images (may take a minute) …")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, n - 1))
    coords = tsne.fit_transform(subset)   # shape: [n, 2]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(coords[:, 0], coords[:, 1], s=10, alpha=0.6, c="steelblue")
    ax.set_title("t-SNE: Image Feature Space\n(images that look alike cluster together)",
                 fontsize=13)
    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")
    ax.axis("equal")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"      Saved → {out_path}")


def main():
    # ── Sanity checks ────────────────────────────────────────────────────────
    for path in (CSV_FILE, PCA_FILE, FILENAMES_FILE):
        if not os.path.exists(path):
            print(f"[ERROR] Required file not found: '{path}'")
            print("  Please run the previous steps first.")
            return

    os.makedirs(PAIRS_DIR, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────────────
    print("[1/3] Loading data …")
    pairs = load_csv(CSV_FILE)
    with open(PCA_FILE, "rb") as f:
        compressed = pickle.load(f)
    with open(FILENAMES_FILE, "rb") as f:
        filenames = pickle.load(f)
    print(f"      {len(pairs)} pairs in CSV, {len(filenames)} images total.")

    # ── Side-by-side pair images ─────────────────────────────────────────────
    n_pairs = min(len(pairs), MAX_PAIRS_TO_VISUALIZE)
    print(f"[2/3] Saving {n_pairs} side-by-side comparison images …")
    for i, row in enumerate(tqdm(pairs[:n_pairs], unit="pair"), start=1):
        out_path = os.path.join(PAIRS_DIR, f"pair_{i:03d}.png")
        save_pair_image(row["image_a"], row["image_b"], row["similarity_score"], out_path)
    print(f"      Saved to {PAIRS_DIR}/")

    # ── t-SNE scatter plot ───────────────────────────────────────────────────
    print("[3/3] Creating t-SNE scatter plot …")
    save_tsne_plot(compressed, filenames, TSNE_PLOT, TSNE_MAX_IMAGES)

    print(f"\n✅ All done!")
    print(f"   Pair images → {PAIRS_DIR}/")
    print(f"   t-SNE plot  → {TSNE_PLOT}")


if __name__ == "__main__":
    main()
