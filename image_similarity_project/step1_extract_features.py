"""
STEP 1 - Extract Features from Images
======================================
What this script does:
  - Looks at every image inside the `images/` folder
  - Uses a pre-trained ResNet50 neural network to turn each image into a
    list of numbers (called a "feature vector") that describes what the
    image looks like
  - Saves those feature vectors and the image file paths to the `pickle/`
    folder so the next step can use them

How to run:
  python step1_extract_features.py

Output files (saved in pickle/):
  features.pickle   - the feature vectors for every image
  filenames.pickle  - the file path of every image (same order as features)
"""

import os
import pickle

import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from torch.nn.functional import normalize
from tqdm import tqdm  # shows a progress bar

# ── Configuration ────────────────────────────────────────────────────────────
IMAGES_DIR = "images"          # folder that contains your images
PICKLE_DIR = "pickle"          # folder where results are saved
FEATURES_FILE = os.path.join(PICKLE_DIR, "features.pickle")
FILENAMES_FILE = os.path.join(PICKLE_DIR, "filenames.pickle")
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# ─────────────────────────────────────────────────────────────────────────────


def get_image_paths(root_dir):
    """Walk root_dir and return a sorted list of all supported image paths."""
    paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if os.path.splitext(fname)[1].lower() in SUPPORTED_EXTENSIONS:
                paths.append(os.path.join(dirpath, fname))
    return sorted(paths)


def build_model(device):
    """Load ResNet50 (pre-trained on ImageNet) and strip the final classifier
    so it outputs a 2048-dimensional feature vector instead of class scores."""
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    # Remove the last fully-connected layer
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.to(device)
    model.eval()
    # Freeze weights – we are only using the model for feature extraction
    for param in model.parameters():
        param.requires_grad = False
    return model


def build_transform():
    """Return the image pre-processing pipeline expected by ResNet50."""
    return transforms.Compose([
        transforms.Resize(232),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def extract_feature(image_path, model, transform, device):
    """
    Open one image, run it through the model, and return a normalised
    1-D feature vector (numpy array of shape [2048]).
    Returns None if the image cannot be opened.
    """
    try:
        img = Image.open(image_path).convert("RGB")
    except (UnidentifiedImageError, OSError):
        return None

    tensor = transform(img).unsqueeze(0).to(device)  # shape: [1, 3, 224, 224]
    with torch.no_grad():
        features = model(tensor)                      # shape: [1, 2048, 1, 1]
    features = features.view(1, -1)                   # shape: [1, 2048]
    features = normalize(features, p=2, dim=1)        # L2-normalise
    return features.squeeze().cpu().numpy()           # shape: [2048]


def main():
    # ── Sanity checks ────────────────────────────────────────────────────────
    if not os.path.isdir(IMAGES_DIR):
        print(f"[ERROR] Images folder not found: '{IMAGES_DIR}'")
        print("  Please create the 'images/' folder and put your images inside it.")
        return

    os.makedirs(PICKLE_DIR, exist_ok=True)

    # ── Collect image paths ──────────────────────────────────────────────────
    print(f"[1/4] Scanning '{IMAGES_DIR}' for images …")
    image_paths = get_image_paths(IMAGES_DIR)
    if not image_paths:
        print("[ERROR] No images found. Supported formats: jpg, jpeg, png, bmp, webp")
        return
    print(f"      Found {len(image_paths)} image(s).")

    # ── Set up device and model ──────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[2/4] Using device: {device}")
    print("      Loading ResNet50 model …")
    model = build_model(device)
    transform = build_transform()

    # ── Extract features ─────────────────────────────────────────────────────
    print("[3/4] Extracting features (this may take a few minutes) …")
    features_list = []
    valid_paths = []
    skipped = 0

    for path in tqdm(image_paths, unit="img"):
        feat = extract_feature(path, model, transform, device)
        if feat is None:
            skipped += 1
            continue
        features_list.append(feat)
        valid_paths.append(path)

    print(f"      Done. Processed {len(valid_paths)} images, skipped {skipped} unreadable file(s).")

    # ── Save to pickle ────────────────────────────────────────────────────────
    print("[4/4] Saving results to pickle/ …")
    with open(FEATURES_FILE, "wb") as f:
        pickle.dump(features_list, f)
    with open(FILENAMES_FILE, "wb") as f:
        pickle.dump(valid_paths, f)

    print(f"\n✅ Done!")
    print(f"   Features saved  → {FEATURES_FILE}")
    print(f"   Filenames saved → {FILENAMES_FILE}")
    print("\nNext step: run  python step2_find_similar_pairs.py")


if __name__ == "__main__":
    main()
