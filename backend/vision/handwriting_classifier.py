"""
vision/handwriting_classifier.py
---------------------------------
Inference module for the trained HOG + LinearSVC handwriting character
classifier. Classifies individual character images as Normal, Corrected,
or Reversal.

Also provides a high-level API to segment characters from a full canvas
image and classify each one, returning an aggregate vision risk assessment.
"""
import os
import numpy as np
import cv2
import joblib
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# ── Module-level cache ────────────────────────────────────────────────────────
_model_cache: Dict[str, Any] = {}


def _load_model():
    """Load the trained HOG+SVM model, scaler, classes, and HOG config."""
    if "clf" not in _model_cache:
        try:
            _model_cache["clf"] = joblib.load(MODELS_DIR / "handwriting_hog_svm.joblib")
            _model_cache["scaler"] = joblib.load(MODELS_DIR / "handwriting_hog_scaler.joblib")
            _model_cache["classes"] = joblib.load(MODELS_DIR / "handwriting_label_classes.joblib")
            _model_cache["hog_config"] = joblib.load(MODELS_DIR / "handwriting_hog_config.joblib")
            _model_cache["available"] = True
        except FileNotFoundError:
            _model_cache["available"] = False
    return _model_cache


def is_model_available() -> bool:
    """Check if the trained handwriting classifier model exists."""
    m = _load_model()
    return m.get("available", False)


def compute_hog_features(img: np.ndarray, config: dict) -> np.ndarray:
    """Compute HOG features matching the training pipeline.

    Fully vectorized numpy — must use identical parameters as
    train_handwriting_cnn.py.
    """
    cell_size = config["cell_size"]
    nbins = config["nbins"]
    block_size = config["block_size"]

    h, w = img.shape
    img_f = img.astype(np.float32) / 255.0

    # Central difference gradients
    gx = np.zeros_like(img_f)
    gy = np.zeros_like(img_f)
    gx[:, 1:-1] = img_f[:, 2:] - img_f[:, :-2]
    gy[1:-1, :] = img_f[2:, :] - img_f[:-2, :]

    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    orientation = np.degrees(np.arctan2(gy, gx)) % 180

    n_cells_y = h // cell_size
    n_cells_x = w // cell_size
    bin_width = 180.0 / nbins

    bin_idx = orientation / bin_width
    lower_bin = np.floor(bin_idx).astype(np.int32) % nbins
    upper_bin = (lower_bin + 1) % nbins
    frac = bin_idx - np.floor(bin_idx)

    lower_weight = magnitude * (1.0 - frac)
    upper_weight = magnitude * frac

    mag_crop_h = n_cells_y * cell_size
    mag_crop_w = n_cells_x * cell_size

    cell_hists = np.zeros((n_cells_y, n_cells_x, nbins), dtype=np.float32)
    for b in range(nbins):
        lower_map = np.where(lower_bin[:mag_crop_h, :mag_crop_w] == b,
                             lower_weight[:mag_crop_h, :mag_crop_w], 0.0)
        upper_map = np.where(upper_bin[:mag_crop_h, :mag_crop_w] == b,
                             upper_weight[:mag_crop_h, :mag_crop_w], 0.0)
        combined = lower_map + upper_map
        cell_hists[:, :, b] = combined.reshape(
            n_cells_y, cell_size, n_cells_x, cell_size
        ).sum(axis=(1, 3))

    n_blocks_y = n_cells_y - block_size + 1
    n_blocks_x = n_cells_x - block_size + 1
    eps = 1e-6

    if n_blocks_y <= 0 or n_blocks_x <= 0:
        return cell_hists.flatten()

    blocks = np.lib.stride_tricks.sliding_window_view(
        cell_hists, (block_size, block_size, nbins)
    ).reshape(n_blocks_y, n_blocks_x, -1)

    norms = np.sqrt(np.sum(blocks ** 2, axis=2, keepdims=True) + eps)
    normalized = blocks / norms

    return normalized.flatten()




def classify_character(char_img: np.ndarray) -> Dict[str, Any]:
    """Classify a single character image.

    Args:
        char_img: Grayscale image (any size) of a single handwritten character.
                  Should be white-on-black (ink = high intensity).

    Returns:
        dict with keys: 'label', 'confidence', 'decision_scores'
    """
    m = _load_model()
    if not m.get("available"):
        return {"label": "Unknown", "confidence": 0.0, "model_available": False}

    clf = m["clf"]
    scaler = m["scaler"]
    classes = m["classes"]
    config = m["hog_config"]
    img_size = config["img_size"]

    # Resize to match training dimensions
    resized = cv2.resize(char_img, (img_size, img_size))

    # Ensure white-on-black
    if np.mean(resized) > 128:
        resized = cv2.bitwise_not(resized)

    feat = compute_hog_features(resized, config)
    feat_scaled = scaler.transform(feat.reshape(1, -1))

    pred_idx = int(clf.predict(feat_scaled)[0])
    label = classes[pred_idx]

    # LinearSVC decision function gives distances to hyperplanes
    decision = clf.decision_function(feat_scaled)[0]
    # Convert decision scores to pseudo-probabilities via softmax
    exp_scores = np.exp(decision - np.max(decision))
    probs = exp_scores / np.sum(exp_scores)

    confidence = float(probs[pred_idx])

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "probabilities": {classes[i]: round(float(probs[i]), 4) for i in range(len(classes))},
        "model_available": True,
    }


def segment_characters(binary_img: np.ndarray, min_area: int = 30) -> List[np.ndarray]:
    """Segment individual characters from a binary (thresholded) canvas image.

    Uses contour detection to find connected components, filters by area,
    and extracts bounding-box crops sorted left-to-right.

    Args:
        binary_img: Thresholded grayscale image (ink = white on black background)
        min_area: Minimum contour area to consider as a character

    Returns:
        List of cropped character images (grayscale, white-on-black)
    """
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter and sort contours by x-position (left to right)
    char_regions = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 3 or h < 3:
            continue
        char_regions.append((x, y, w, h))

    # Sort left-to-right
    char_regions.sort(key=lambda r: r[0])

    # Extract character crops with a small padding
    char_images = []
    img_h, img_w = binary_img.shape
    for x, y, w, h in char_regions:
        pad = max(2, int(0.1 * max(w, h)))
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(img_w, x + w + pad)
        y1 = min(img_h, y + h + pad)
        crop = binary_img[y0:y1, x0:x1]

        # Make square by padding the shorter dimension
        ch, cw = crop.shape
        side = max(ch, cw)
        square = np.zeros((side, side), dtype=np.uint8)
        dy = (side - ch) // 2
        dx = (side - cw) // 2
        square[dy:dy + ch, dx:dx + cw] = crop
        char_images.append(square)

    return char_images


def classify_canvas_characters(
    binary_img: np.ndarray,
    min_area: int = 30,
) -> Dict[str, Any]:
    """Segment and classify all characters in a canvas image.

    Args:
        binary_img: Thresholded grayscale image (ink = white on black)
        min_area: Minimum contour area for character detection

    Returns:
        dict with:
          - 'total_characters': int
          - 'per_character': list of classification results
          - 'summary': counts per class
          - 'reversal_ratio': fraction classified as Reversal
          - 'corrected_ratio': fraction classified as Corrected
          - 'vision_risk_score': 0–100 score based on reversal/correction prevalence
    """
    if not is_model_available():
        return {
            "total_characters": 0,
            "per_character": [],
            "summary": {},
            "reversal_ratio": 0.0,
            "corrected_ratio": 0.0,
            "vision_risk_score": 0,
            "model_available": False,
        }

    char_images = segment_characters(binary_img, min_area=min_area)

    if not char_images:
        return {
            "total_characters": 0,
            "per_character": [],
            "summary": {"Normal": 0, "Corrected": 0, "Reversal": 0},
            "reversal_ratio": 0.0,
            "corrected_ratio": 0.0,
            "vision_risk_score": 0,
            "model_available": True,
        }

    per_char = []
    for i, char_img in enumerate(char_images):
        result = classify_character(char_img)
        result["char_index"] = i
        per_char.append(result)

    total = len(per_char)
    counts = {"Normal": 0, "Corrected": 0, "Reversal": 0}
    for r in per_char:
        label = r.get("label", "Unknown")
        if label in counts:
            counts[label] += 1

    reversal_ratio = counts["Reversal"] / total if total > 0 else 0.0
    corrected_ratio = counts["Corrected"] / total if total > 0 else 0.0

    # Vision risk score (0–100)
    # Reversals are strong indicators, corrections are mild indicators
    # Weighted: reversal=3x, corrected=1x
    risk_raw = (counts["Reversal"] * 3 + counts["Corrected"] * 1) / (total * 3) * 100
    vision_risk_score = int(min(round(risk_raw), 100))

    return {
        "total_characters": total,
        "per_character": per_char,
        "summary": counts,
        "reversal_ratio": round(reversal_ratio, 4),
        "corrected_ratio": round(corrected_ratio, 4),
        "vision_risk_score": vision_risk_score,
        "model_available": True,
    }
