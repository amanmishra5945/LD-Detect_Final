"""
vision/handwriting_analyzer.py
------------------------------
Computer Vision & Kinematic Handwriting Analysis Engine.
Combines OpenCV image processing (contours, Hu moments, projection profiles, density)
with dynamic pen stroke trajectory kinematics (speed, smoothness, pauses, baseline drift).

Identifies structural letter-form mismatches (e.g. 'b' vs 'd' loop orientation)
and computes explainable motor-control and spatial-organization indicators.
"""
import base64
import io
import math
from typing import List, Dict, Any, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from vision.handwriting_classifier import classify_canvas_characters, is_model_available


def decode_or_render_canvas(
    strokes: List[List[Dict[str, Any]]],
    image_base64: Optional[str] = None,
    width: int = 800,
    height: int = 340,
) -> np.ndarray:
    """
    Renders strokes onto a high-contrast grayscale image using OpenCV, or decodes
    an existing base64 PNG dataURL. Guarantees a clean, standardized image for CV analysis.
    """
    if image_base64 and "," in image_base64:
        try:
            raw_b64 = image_base64.split(",", 1)[1]
            img_bytes = base64.b64decode(raw_b64)
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("L")
            img = np.array(pil_img)
            # Invert so ink is white (255) on black (0) background
            if np.mean(img) > 128:
                img = cv2.bitwise_not(img)
            return cv2.resize(img, (width, height))
        except Exception:
            pass  # fallback to rendering from strokes

    # Render from stroke coordinates
    canvas = np.zeros((height, width), dtype=np.uint8)
    for stroke in strokes:
        if len(stroke) < 2:
            if len(stroke) == 1:
                pt = (int(stroke[0]["x"]), int(stroke[0]["y"]))
                cv2.circle(canvas, pt, 2, 255, -1)
            continue
        pts = np.array([[int(p["x"]), int(p["y"])] for p in stroke], dtype=np.int32)
        cv2.polylines(canvas, [pts], isClosed=False, color=255, thickness=3, lineType=cv2.LINE_AA)

    return canvas


def compute_hu_moments(binary_img: np.ndarray) -> List[float]:
    """Calculate scale and rotation invariant log-transformed Hu Moments."""
    moments = cv2.moments(binary_img)
    hu = cv2.HuMoments(moments)
    log_hu = []
    for h in hu:
        val = float(h[0])
        # Log scale with sign preservation
        if abs(val) > 1e-30:
            log_hu.append(-1.0 * math.copysign(1.0, val) * math.log10(abs(val)))
        else:
            log_hu.append(0.0)
    return log_hu


def detect_loop_orientation(binary_img: np.ndarray, target_letter: str) -> Optional[str]:
    """
    Checks horizontal asymmetry for reversible letter forms like 'b' vs 'd' or 'p' vs 'q'.
    In 'b': vertical stem on the left, rounded loop on the right.
    In 'd': vertical stem on the right, rounded loop on the left.
    """
    target = (target_letter or "").strip().lower()
    if target not in ("b", "d", "p", "q"):
        return None

    # Find the main contour
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    main_cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(main_cnt)
    if w < 10 or h < 15:
        return None

    crop = binary_img[y:y+h, x:x+w]
    h_mid = h // 2
    w_mid = w // 2

    # Top half mass
    top_left = float(np.sum(crop[:h_mid, :w_mid] > 0))
    top_right = float(np.sum(crop[:h_mid, w_mid:] > 0))
    top_total = top_left + top_right
    top_left_ratio = top_left / (top_total + 1e-5) if top_total > 0 else 0.5

    # Bottom half mass
    bot_left = float(np.sum(crop[h_mid:, :w_mid] > 0))
    bot_right = float(np.sum(crop[h_mid:, w_mid:] > 0))
    bot_total = bot_left + bot_right
    bot_left_ratio = bot_left / (bot_total + 1e-5) if bot_total > 0 else 0.5

    if target == "b":
        # 'b' vertical ascender stem is in upper-left; if stem is on upper-right, it's 'd'
        if top_left_ratio < 0.45 and top_right > top_left * 1.2:
            return "Reversal detected: letter was written as 'd' (ascender stem on right) instead of 'b'"
    elif target == "d":
        # 'd' vertical ascender stem is in upper-right; if stem is on upper-left, it's 'b'
        if top_left_ratio > 0.55 and top_left > top_right * 1.2:
            return "Reversal detected: letter was written as 'b' (ascender stem on left) instead of 'd'"
    elif target == "p":
        # 'p' vertical descender stem is in lower-left; if stem is on lower-right, it's 'q'
        if bot_left_ratio < 0.45 and bot_right > bot_left * 1.2:
            return "Reversal detected: letter was written as 'q' (descender stem on right) instead of 'p'"
    elif target == "q":
        # 'q' vertical descender stem is in lower-right; if stem is on lower-left, it's 'p'
        if bot_left_ratio > 0.55 and bot_left > bot_right * 1.2:
            return "Reversal detected: letter was written as 'p' (descender stem on left) instead of 'q'"

    return None


def extract_stroke_kinematics(
    strokes: List[List[Dict[str, Any]]],
    canvas_width: int,
    canvas_height: int,
    total_duration_sec: float
) -> Dict[str, Any]:
    """
    Calculates detailed stroke kinematics:
    speed, acceleration/jerk, smoothness, pauses, pen lifts, and baseline variability.
    """
    valid_strokes = []
    for s in strokes:
        if len(s) >= 2:
            path_len = sum(
                math.sqrt(
                    ((s[i]["x"] - s[i-1]["x"]) / canvas_width) ** 2 +
                    ((s[i]["y"] - s[i-1]["y"]) / canvas_height) ** 2
                )
                for i in range(1, len(s))
            )
            if path_len >= 0.002:
                valid_strokes.append(s)

    stroke_count = len(valid_strokes)
    pen_lifts = max(stroke_count - 1, 0)

    point_speeds = []
    turn_angles = []
    boxes = []
    pressure_values = []
    on_surface_time = 0.0

    for stroke in valid_strokes:
        xs = [float(p["x"]) / canvas_width for p in stroke]
        ys = [float(p["y"]) / canvas_height for p in stroke]
        boxes.append({
            "x0": min(xs), "x1": max(xs),
            "y0": min(ys), "y1": max(ys),
            "cy": (min(ys) + max(ys)) / 2.0
        })

        if "t" in stroke[-1] and "t" in stroke[0]:
            on_surface_time += max(float(stroke[-1]["t"]) - float(stroke[0]["t"]), 0.0)

        for p in stroke:
            press = float(p.get("pressure", 0.0) or 0.0)
            if press > 0:
                pressure_values.append(press)

        directions = []
        for i in range(1, len(stroke)):
            dx = (float(stroke[i]["x"]) - float(stroke[i-1]["x"])) / canvas_width
            dy = (float(stroke[i]["y"]) - float(stroke[i-1]["y"])) / canvas_height
            dt = max(float(stroke[i].get("t", 0)) - float(stroke[i-1].get("t", 0)), 0.005)
            dist = math.sqrt(dx*dx + dy*dy)
            if dist > 0:
                point_speeds.append(dist / dt)
                directions.append(math.atan2(dy, dx))

        for a, b in zip(directions, directions[1:]):
            delta = abs((b - a + math.pi) % (2 * math.pi) - math.pi)
            turn_angles.append(delta / math.pi)

    if point_speeds:
        speeds_arr = np.asarray(point_speeds)
        lo, hi = np.percentile(speeds_arr, [10, 90])
        robust_speeds = np.clip(speeds_arr, lo, hi)
        median_speed = float(np.median(robust_speeds))
        speed_irregularity = float(np.std(robust_speeds) / (np.mean(robust_speeds) + 1e-6))
        turn_irregularity = float(np.mean(turn_angles)) if turn_angles else 0.25
        stroke_smoothness = round(
            float(np.clip(1.0 - 0.38 * speed_irregularity - 0.22 * turn_irregularity, 0.30, 0.99)),
            3
        )
    else:
        median_speed = 0.0
        speed_irregularity = 1.0
        stroke_smoothness = 0.60

    # Cluster overlapping horizontal strokes to estimate letter units
    clusters = []
    current_cluster = None
    for b in sorted(boxes, key=lambda item: item["x0"]):
        if current_cluster is None or b["x0"] > current_cluster["x1"] + 0.015:
            current_cluster = dict(b)
            clusters.append(current_cluster)
        else:
            current_cluster["x0"] = min(current_cluster["x0"], b["x0"])
            current_cluster["x1"] = max(current_cluster["x1"], b["x1"])
            current_cluster["y0"] = min(current_cluster["y0"], b["y0"])
            current_cluster["y1"] = max(current_cluster["y1"], b["y1"])

    heights = [c["y1"] - c["y0"] for c in clusters if (c["y1"] - c["y0"]) > 0.01]
    letter_size_cv = float(np.std(heights) / (np.mean(heights) + 1e-6)) if len(heights) > 2 else 0.25

    # Horizontal gaps
    gaps = [clusters[i]["x0"] - clusters[i-1]["x1"] for i in range(1, len(clusters))]
    positive_gaps = [g for g in gaps if g > 0.002]
    spacing_cv = float(np.std(positive_gaps) / (np.mean(positive_gaps) + 1e-6)) if len(positive_gaps) > 2 else 0.30

    # Baseline deviation
    base_ys = [c["y1"] for c in clusters]
    baseline_deviation = float(np.std(base_ys)) if len(base_ys) > 2 else 0.02

    # Pause analysis
    ordered_strokes = sorted(valid_strokes, key=lambda s: float(s[0].get("t", 0.0)))
    pauses = [
        max(float(ordered_strokes[i][0].get("t", 0)) - float(ordered_strokes[i-1][-1].get("t", 0)), 0.0)
        for i in range(1, len(ordered_strokes))
    ]
    meaningful_pauses = [p for p in pauses if p >= 0.4]
    long_pauses = [p for p in pauses if p >= 1.5]

    return {
        "stroke_count": stroke_count,
        "pen_lifts": pen_lifts,
        "median_speed": round(median_speed * 100, 2),
        "stroke_smoothness": stroke_smoothness,
        "speed_irregularity": round(speed_irregularity, 3),
        "letter_size_cv": round(letter_size_cv, 3),
        "spacing_cv": round(spacing_cv, 3),
        "baseline_deviation": round(baseline_deviation, 4),
        "meaningful_pauses": len(meaningful_pauses),
        "long_pauses": len(long_pauses),
        "mean_pause_sec": round(float(np.mean(meaningful_pauses)), 2) if meaningful_pauses else 0.0,
        "clusters_count": len(clusters),
        "on_surface_ratio": round(min(on_surface_time / max(total_duration_sec, 0.1), 1.0), 3)
    }


def verify_handwritten_target(
    binary_img: np.ndarray,
    target_text: str,
    letter_mismatch_flag: Optional[str] = None
) -> Dict[str, Any]:
    """
    Rigorously verifies whether the drawn canvas content matches the requested
    target letter, word, or phrase.
    """
    target = (target_text or "").strip()
    if not target:
        return {"is_matched": True, "accuracy_pct": 100, "details": "No target specified.", "reasons": []}

    ink_pixels = int(np.sum(binary_img > 0))
    if ink_pixels < 30:
        return {
            "is_matched": False,
            "accuracy_pct": 0,
            "detected_chars": 0,
            "expected_chars": len(target.replace(" ", "")),
            "reasons": ["Canvas is blank or writing has insufficient ink."]
        }

    clean_target = target.replace(" ", "")
    n_expected = len(clean_target)

    # 1. Topological loop / hole inspection
    HOLE_1 = set("abdeopqADOPQR0469")
    HOLE_2 = set("B8")
    exp_holes = sum(2 if ch in HOLE_2 else (1 if ch in HOLE_1 else 0) for ch in clean_target)

    cnts, hier = cv2.findContours(binary_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    drawn_holes = 0
    if hier is not None:
        for h in hier[0]:
            if h[3] != -1:  # interior hole
                drawn_holes += 1

    # 2. Extracted character units (filtered by min area)
    ext_cnts, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_cnts = [c for c in ext_cnts if cv2.contourArea(c) >= 35]
    n_detected = len(valid_cnts)

    # Character count agreement score
    if n_expected == 1:
        if n_detected == 1:
            count_score = 1.0
        elif n_detected == 2:
            count_score = 0.85
        elif n_detected == 3:
            count_score = 0.50
        else:
            count_score = max(0.10, 1.0 - (n_detected - 1) * 0.25)
    else:
        char_diff = abs(n_detected - n_expected)
        count_score = max(0.0, 1.0 - (char_diff / max(n_expected, 1)))

    # Topological hole agreement score
    hole_diff = abs(drawn_holes - exp_holes)
    if exp_holes == 0 and drawn_holes > 1:
        hole_score = 0.30
    elif exp_holes > 0 and drawn_holes == 0:
        hole_score = 0.40
    else:
        hole_score = max(0.0, 1.0 - hole_diff * 0.35)

    # 3. Structural template comparison via bilateral distance transform
    scale = 1.8 if n_expected <= 2 else (1.4 if n_expected <= 6 else 1.0)
    thick = 3
    tmpl = np.zeros((120, max(140, n_expected * 55)), dtype=np.uint8)
    cv2.putText(tmpl, target, (15, 80), cv2.FONT_HERSHEY_SIMPLEX, scale, 255, thick, cv2.LINE_AA)

    def get_crop(img):
        pts = np.argwhere(img > 0)
        if len(pts) == 0:
            return np.zeros((64, 64), dtype=np.uint8)
        y0, x0 = pts.min(axis=0)
        y1, x1 = pts.max(axis=0) + 1
        return img[y0:y1, x0:x1]

    c_drawn = get_crop(binary_img)
    c_tmpl = get_crop(tmpl)

    h_fixed = 80
    def norm_h(c):
        h, w = c.shape
        nw = max(int(w * (h_fixed / max(h, 1))), 1)
        return cv2.resize(c, (nw, h_fixed))

    n_d = norm_h(c_drawn)
    n_t = norm_h(c_tmpl)

    w_ratio = min(n_d.shape[1], n_t.shape[1]) / max(n_d.shape[1], n_t.shape[1])

    max_w = max(n_d.shape[1], n_t.shape[1])
    def pad_w(img):
        out = np.zeros((h_fixed, max_w), dtype=np.uint8)
        dx = (max_w - img.shape[1]) // 2
        out[:, dx:dx+img.shape[1]] = img
        return out

    p_d = pad_w(n_d)
    p_t = pad_w(n_t)

    dist_map_t = cv2.distanceTransform(cv2.bitwise_not(p_t), cv2.DIST_L2, 3)
    pts_d = p_d > 0
    d1 = np.mean(dist_map_t[pts_d]) if np.sum(pts_d) > 0 else 50.0

    dist_map_d = cv2.distanceTransform(cv2.bitwise_not(p_d), cv2.DIST_L2, 3)
    pts_t = p_t > 0
    d2 = np.mean(dist_map_d[pts_t]) if np.sum(pts_t) > 0 else 50.0

    sym_d = (d1 + d2) / 2.0
    chamfer_score = float(np.exp(-sym_d / 12.0))

    try:
        shape_diff = cv2.matchShapes(n_d, n_t, cv2.CONTOURS_MATCH_I1, 0)
        shape_sim = float(np.exp(-min(shape_diff, 5.0)))
    except Exception:
        shape_sim = 0.50

    raw_match = (
        0.35 * chamfer_score +
        0.25 * count_score +
        0.15 * hole_score +
        0.15 * shape_sim +
        0.10 * w_ratio
    )

    has_reversal = bool(letter_mismatch_flag)
    if has_reversal:
        raw_match *= 0.50

    accuracy_pct = int(np.clip(round(raw_match * 100), 5, 98))
    is_matched = (accuracy_pct >= 55) and not has_reversal

    reasons = []
    if has_reversal:
        reasons.append(f"Reversal detected: letter was written in reversed orientation.")
    elif accuracy_pct < 45:
        reasons.append(f"Drawn shape does not match target '{target}' (illegible formation or incorrect characters).")
    elif count_score < 0.5:
        reasons.append(f"Character count mismatch: Expected {n_expected} letters, but detected {n_detected} stroke units.")

    return {
        "is_matched": is_matched,
        "accuracy_pct": accuracy_pct,
        "detected_chars": n_detected,
        "expected_chars": n_expected,
        "drawn_holes": drawn_holes,
        "expected_holes": exp_holes,
        "has_reversal": has_reversal,
        "reasons": reasons
    }


def analyze_handwriting_task(
    strokes: List[List[Dict[str, Any]]],
    target_text: str,
    age: int,
    total_duration_sec: float,
    correction_count: int = 0,
    image_base64: Optional[str] = None,
    canvas_width: int = 800,
    canvas_height: int = 340,
) -> Dict[str, Any]:
    """
    Comprehensive analysis combining Computer Vision (OpenCV) and Stroke Kinematics.
    Rigorously verifies target content match, kinematics, and character formation.
    """
    duration = max(float(total_duration_sec or 0.0), 0.1)
    target = target_text.strip()
    target_letter_count = max(len(target), 1)

    # 1. Computer Vision processing via OpenCV
    cv_img = decode_or_render_canvas(strokes, image_base64, canvas_width, canvas_height)
    
    # Thresholding & morphological cleanup
    _, thresh = cv2.threshold(cv_img, 30, 255, cv2.THRESH_BINARY)
    ink_pixels = int(np.sum(thresh > 0))
    total_pixels = canvas_width * canvas_height
    fill_density = ink_pixels / float(total_pixels)

    # Contours analysis
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_contours = len(contours)

    hu_moments = compute_hu_moments(thresh) if ink_pixels > 20 else [0.0]*7

    # Check for letter-form mismatch if target is a confusable letter
    letter_mismatch_flag = None
    if len(target) == 1 and target.isalpha():
        letter_mismatch_flag = detect_loop_orientation(thresh, target)

    # 2. Kinematics analysis
    kinematics = extract_stroke_kinematics(strokes, canvas_width, canvas_height, duration)
    writing_speed_cps = round(target_letter_count / duration, 3)

    # 3. Rigorous Target Word / Letter Content Verification
    verification = verify_handwritten_target(thresh, target, letter_mismatch_flag)
    visual_similarity = round(verification["accuracy_pct"] / 100.0, 2)

    observations = []
    observations.extend(verification.get("reasons", []))

    if kinematics["stroke_count"] < 2 and target_letter_count > 1:
        observations.append("Very few strokes recorded for this task.")
    elif kinematics["stroke_smoothness"] >= 0.75:
        observations.append("Stroke execution was smooth and controlled.")
    elif kinematics["stroke_smoothness"] < 0.55:
        observations.append("Notable stroke jerkiness or motor irregularity observed.")

    if letter_mismatch_flag:
        observations.append(letter_mismatch_flag)

    if kinematics["baseline_deviation"] > 0.06:
        observations.append("Baseline alignment exhibited noticeable vertical drift.")

    if kinematics["letter_size_cv"] > 0.60:
        observations.append("Character size varied substantially across written elements.")

    if correction_count > 0:
        observations.append(f"{correction_count} correction(s)/erasure(s) performed.")

    # 4. HOG+SVM character classification (if trained model is available)
    char_classification = None
    if is_model_available() and ink_pixels > 20:
        char_classification = classify_canvas_characters(thresh, min_area=30)
        if char_classification and char_classification.get("model_available"):
            summary = char_classification.get("summary", {})
            total_chars = char_classification.get("total_characters", 0)
            reversals = summary.get("Reversal", 0)
            corrected = summary.get("Corrected", 0)

            if reversals > 0 and total_chars > 0:
                observations.append(
                    f"Vision model detected {reversals}/{total_chars} character(s) "
                    f"with reversed letter-form patterns."
                )
            if corrected > 0 and total_chars > 0:
                observations.append(
                    f"Vision model detected {corrected}/{total_chars} character(s) "
                    f"with self-correction patterns."
                )

    return {
        "target": target,
        "metrics": {
            "stroke_count": kinematics["stroke_count"],
            "pen_lifts": kinematics["pen_lifts"],
            "duration_sec": round(duration, 2),
            "writing_speed_cps": writing_speed_cps,
            "stroke_smoothness": kinematics["stroke_smoothness"],
            "speed_irregularity": kinematics["speed_irregularity"],
            "median_speed": kinematics["median_speed"],
            "letter_size_cv": kinematics["letter_size_cv"],
            "spacing_cv": kinematics["spacing_cv"],
            "baseline_deviation": kinematics["baseline_deviation"],
            "pauses_count": kinematics["meaningful_pauses"],
            "long_pauses": kinematics["long_pauses"],
            "mean_pause_sec": kinematics["mean_pause_sec"],
            "ink_density": round(fill_density, 5),
            "num_contours": num_contours,
            "corrections": correction_count
        },
        "cv_features": {
            "fill_density": round(fill_density, 5),
            "contour_count": num_contours,
            "hu_moments": [round(h, 4) for h in hu_moments[:4]]
        },
        "visual_similarity": visual_similarity,
        "verification": verification,
        "letter_form_flag": letter_mismatch_flag,
        "char_classification": char_classification,
        "observations": observations
    }
