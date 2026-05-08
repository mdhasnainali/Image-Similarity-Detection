# Image Similarity Finder

Given a folder of images, this project automatically finds which images look
the most alike and saves the results in a spreadsheet (CSV file) you can open
in Excel or Google Sheets.

---

## What you need before starting

| Requirement | How to check |
|---|---|
| Python 3.8 or newer | Open a terminal and type `python --version` |
| pip (Python package installer) | Type `pip --version` |

No other knowledge is required. Just follow the steps below.

---

## Folder structure

```
image_similarity_project/
│
├── images/               ← PUT YOUR IMAGES HERE
│
├── pickle/               ← intermediate files (created automatically)
├── output/               ← results go here (created automatically)
│   ├── similar_pairs.csv ← open this in Excel to see the results
│   ├── tsne_plot.png     ← scatter plot of all images
│   └── pairs/            ← side-by-side comparison images
│
├── step1_extract_features.py
├── step2_find_similar_pairs.py
├── step3_visualize.py
└── requirements.txt
```

---

## Step-by-step guide

### 1. Install the required libraries (do this once)

Open a terminal, navigate to this folder, and run:

```bash
pip install -r requirements.txt
```

This downloads all the Python libraries the scripts need. It may take a few
minutes the first time.

---

### 2. Add your images

Copy all the images you want to compare into the `images/` folder.

Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`

You can have sub-folders inside `images/` — the script will find all images
automatically.

---

### 3. Run Step 1 — Extract features

```bash
python step1_extract_features.py
```

**What happens:**
- The script looks at every image in `images/`
- It uses a neural network (ResNet50) to turn each image into a list of
  numbers that describes what the image looks like
- The results are saved to `pickle/features.pickle` and
  `pickle/filenames.pickle`

**How long does it take?**
About 1–2 seconds per image on a CPU. For 1 000 images expect ~20 minutes.
If you have an NVIDIA GPU it will be much faster automatically.

---

### 4. Run Step 2 — Find similar pairs

```bash
python step2_find_similar_pairs.py
```

**What happens:**
- Loads the features from Step 1
- Compresses them with PCA (saves to `pickle/pca_features.pickle`)
- Builds a fast search index with FAISS (saves to `pickle/faiss_index.pickle`)
- Finds the most similar pair for every image
- Writes the top 100 most similar pairs to `output/similar_pairs.csv`

**Understanding the CSV columns:**

| Column | Meaning |
|---|---|
| rank | 1 = most similar pair, 2 = second most similar, … |
| image_a | Path to the first image |
| image_b | Path to the second image |
| distance | How far apart the images are (lower = more similar) |
| similarity_score | 0.0 to 1.0 — closer to 1.0 means more similar |

---

### 5. Run Step 3 — Visualize

```bash
python step3_visualize.py
```

**What happens:**
- Creates side-by-side comparison images for the top 20 pairs and saves them
  to `output/pairs/`
- Creates a t-SNE scatter plot (`output/tsne_plot.png`) where images that
  look alike appear close together

---

## Troubleshooting

**"No images found"**
Make sure your images are inside the `images/` folder (not somewhere else).

**"Required file not found: pickle/features.pickle"**
You skipped Step 1. Run `python step1_extract_features.py` first.

**Script is very slow**
This is normal on a CPU for large datasets. The neural network runs faster
on a GPU. You can also reduce the number of images.

**An image is skipped with a warning**
The image file may be corrupted or not a real image. The script skips it
and continues with the rest.

---

## How it works (simple explanation)

1. **Feature extraction** — A neural network trained on millions of photos
   looks at each image and produces a "fingerprint" (a list of 2048 numbers).
   Images that look similar have similar fingerprints.

2. **PCA compression** — The 2048 numbers are compressed to 200 numbers.
   This makes the search much faster while keeping almost all the information.

3. **FAISS search** — FAISS is a library that can search through millions of
   fingerprints in milliseconds to find the closest matches.

4. **CSV output** — The closest pairs are ranked and saved to a spreadsheet.
