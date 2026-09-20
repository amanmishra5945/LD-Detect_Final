"""
train_handwriting_cnn.py
------------------------
Trains a handwriting character classifier on the Kaggle "Gambo" Dyslexia
Handwriting Image Dataset using vectorized HOG-like gradient histogram
features + a scikit-learn LinearSVC classifier.

All feature extraction is fully vectorized with numpy — no Python-level
loops over pixels or cells.

Run:  python train_handwriting_cnn.py
Output: models/handwriting_hog_svm.joblib
        models/handwriting_hog_scaler.joblib
        models/handwriting_label_classes.joblib
        models/handwriting_hog_config.joblib
"""
import sys
import time
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import cv2

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ── Dataset location ──────────────────────────────────────────────────────────
DATASET_ROOT = Path(r"C:\Users\amanm\Downloads\archive\Dataset Dyslexia_Password WanAsy321\Gambo")

CLASSES = ["Corrected", "Normal", "Reversal"]
IMG_SIZE = 32  # resize all images to 32x32

# ── HOG-like parameters ──────────────────────────────────────────────────────
CELL_SIZE = 8     # pixels per cell
NBINS = 9         # orientation bins (0–180°, unsigned gradients)
BLOCK_SIZE = 2    # cells per block for normalization


def compute_hog_features(img: np.ndarray) -> np.ndarray:
    """Compute HOG features using fully vectorized numpy operations.

    No Python-level loops over pixels or cells — pure array operations.
    """
    h, w = img.shape
    img_f = img.astype(np.float32) / 255.0

    # Gradients via central differences
    gx = np.zeros_like(img_f)
    gy = np.zeros_like(img_f)
    gx[:, 1:-1] = img_f[:, 2:] - img_f[:, :-2]
    gy[1:-1, :] = img_f[2:, :] - img_f[:-2, :]

    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    orientation = np.degrees(np.arctan2(gy, gx)) % 180  # unsigned [0, 180)

    n_cells_y = h // CELL_SIZE
    n_cells_x = w // CELL_SIZE
    bin_width = 180.0 / NBINS

    # Compute bin indices and fractional weights for bilinear interpolation
    bin_idx = orientation / bin_width
    lower_bin = np.floor(bin_idx).astype(np.int32) % NBINS
    upper_bin = (lower_bin + 1) % NBINS
    frac = bin_idx - np.floor(bin_idx)

    lower_weight = magnitude * (1.0 - frac)
    upper_weight = magnitude * frac

    # Build per-bin magnitude images, then sum within cells via reshape
    cell_hists = np.zeros((n_cells_y, n_cells_x, NBINS), dtype=np.float32)

    # Crop to exact cell grid
    mag_crop_h = n_cells_y * CELL_SIZE
    mag_crop_w = n_cells_x * CELL_SIZE

    for b in range(NBINS):
        # Lower bin contribution
        lower_map = np.where(lower_bin[:mag_crop_h, :mag_crop_w] == b,
                             lower_weight[:mag_crop_h, :mag_crop_w], 0.0)
        # Upper bin contribution
        upper_map = np.where(upper_bin[:mag_crop_h, :mag_crop_w] == b,
                             upper_weight[:mag_crop_h, :mag_crop_w], 0.0)
        combined = lower_map + upper_map
        # Reshape to (n_cells_y, CELL_SIZE, n_cells_x, CELL_SIZE), then sum over cell pixels
        cell_hists[:, :, b] = combined.reshape(
            n_cells_y, CELL_SIZE, n_cells_x, CELL_SIZE
        ).sum(axis=(1, 3))

    # Block normalization (L2-norm over overlapping 2×2 cell blocks)
    n_blocks_y = n_cells_y - BLOCK_SIZE + 1
    n_blocks_x = n_cells_x - BLOCK_SIZE + 1
    eps = 1e-6

    if n_blocks_y <= 0 or n_blocks_x <= 0:
        return cell_hists.flatten()

    blocks = np.lib.stride_tricks.sliding_window_view(
        cell_hists, (BLOCK_SIZE, BLOCK_SIZE, NBINS)
    ).reshape(n_blocks_y, n_blocks_x, -1)

    norms = np.sqrt(np.sum(blocks ** 2, axis=2, keepdims=True) + eps)
    normalized = blocks / norms

    return normalized.flatten()


def load_images_from_split(split: str, max_per_class: int = 0):
    """Load images and extract HOG features from a Train or Test split."""
    features = []
    labels = []

    for cls in CLASSES:
        cls_dir = DATASET_ROOT / split / cls
        if not cls_dir.exists():
            print(f"  WARNING: {cls_dir} not found, skipping")
            continue

        files = [f for f in cls_dir.iterdir()
                 if f.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp')]
        if max_per_class > 0 and len(files) > max_per_class:
            rng = np.random.default_rng(42)
            indices = rng.choice(len(files), max_per_class, replace=False)
            files = [files[i] for i in indices]

        loaded = 0
        t0 = time.time()
        for fpath in files:
            try:
                img = np.array(Image.open(fpath).convert("L"))
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                feat = compute_hog_features(img)
                features.append(feat)
                labels.append(cls)
                loaded += 1
            except Exception:
                pass

            # Progress indicator every 2000 images
            if loaded % 2000 == 0 and loaded > 0:
                elapsed = time.time() - t0
                rate = loaded / elapsed
                print(f"    {split}/{cls}: {loaded}/{len(files)} "
                      f"({rate:.0f} img/s)")

        elapsed = time.time() - t0
        print(f"  {split}/{cls}: loaded {loaded} images ({elapsed:.1f}s)")

    return np.array(features), np.array(labels)


def train():
    """Full training pipeline."""
    print("=" * 60)
    print("  Handwriting Character Classifier — HOG + LinearSVC")
    print("=" * 60)
    print(f"\nDataset root: {DATASET_ROOT}")
    print(f"Image resize: {IMG_SIZE}x{IMG_SIZE}")
    print(f"HOG params: cell={CELL_SIZE}px, bins={NBINS}, "
          f"block={BLOCK_SIZE}x{BLOCK_SIZE} cells")

    if not DATASET_ROOT.exists():
        print(f"\nERROR: Dataset not found at {DATASET_ROOT}")
        sys.exit(1)

    # ── Load training data ────────────────────────────────────────────────
    print("\n[1/4] Loading training images + extracting HOG features...")
    t0 = time.time()
    X_train, y_train = load_images_from_split("Train", max_per_class=12000)
    print(f"  Total training samples: {len(X_train)}  "
          f"({time.time() - t0:.1f}s)")
    print(f"  Feature vector length: {X_train.shape[1]}")

    # ── Load test data ────────────────────────────────────────────────────
    print("\n[2/4] Loading test images + extracting HOG features...")
    t0 = time.time()
    X_test, y_test = load_images_from_split("Test")
    print(f"  Total test samples: {len(X_test)}  ({time.time() - t0:.1f}s)")

    # ── Encode labels ─────────────────────────────────────────────────────
    le = LabelEncoder()
    le.fit(CLASSES)
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)

    # ── Scale features ────────────────────────────────────────────────────
    print("\n[3/4] Scaling features and training LinearSVC...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    t0 = time.time()
    clf = LinearSVC(
        C=1.0,
        max_iter=3000,
        class_weight="balanced",
        random_state=42,
        dual="auto",
    )
    clf.fit(X_train_s, y_train_enc)
    train_time = time.time() - t0
    print(f"  Training completed in {train_time:.1f}s")

    # ── Evaluate ──────────────────────────────────────────────────────────
    print("\n[4/4] Evaluating on test set...")
    preds = clf.predict(X_test_s)
    acc = accuracy_score(y_test_enc, preds)

    print(f"\n  Test Accuracy: {acc:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test_enc, preds, target_names=le.classes_))
    print("  Confusion Matrix (rows=true, cols=predicted):")
    import pandas as pd
    cm = confusion_matrix(y_test_enc, preds)
    print(pd.DataFrame(cm, index=le.classes_, columns=le.classes_))

    # ── Save artifacts ────────────────────────────────────────────────────
    model_path = MODELS_DIR / "handwriting_hog_svm.joblib"
    scaler_path = MODELS_DIR / "handwriting_hog_scaler.joblib"
    classes_path = MODELS_DIR / "handwriting_label_classes.joblib"

    joblib.dump(clf, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(le.classes_.tolist(), classes_path)

    hog_config = {
        "img_size": IMG_SIZE,
        "cell_size": CELL_SIZE,
        "nbins": NBINS,
        "block_size": BLOCK_SIZE,
    }
    joblib.dump(hog_config, MODELS_DIR / "handwriting_hog_config.joblib")

    print(f"\n  Model saved to:   {model_path}")
    print(f"  Scaler saved to:  {scaler_path}")
    print(f"  Classes saved to: {classes_path}")
    print("=" * 60)

    return acc


if __name__ == "__main__":
    train()
