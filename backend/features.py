"""
features.py
Converts raw front-end assessment payloads into the exact feature vectors
the trained models expect — the same 11 columns (per disorder) present in
final_dyslexia_dataset_1000.xlsx / final_dysgraphia_dataset_1000.xlsx.

NOTE on the handwriting features: the real dataset's Stroke_Count /
Pen_Lift_Count ranges (~100-250 / ~8-90) imply it was captured from a
longer, tablet/stylus-recorded writing task, not a 3-word mouse scribble.
A browser <canvas> + mouse is a rough proxy for real handwriting capture,
so the frontend now asks for a longer copy-sentence and the geometric
features below (letter size / spacing / baseline deviation) are scaled by
calibration constants to land in the same numeric neighborhood as the
training data. This is flagged in the README as a demo-grade
approximation — a production deployment would use a stylus/tablet with
consistent DPI so no calibration guesswork is needed.
"""
import re
from difflib import SequenceMatcher

import numpy as np

# ---- calibration constants for pixel-space -> dataset-scale geometric features ----
# (tunable; see README "Calibration" note)
LETTER_SIZE_SCALE = 12.0     # px stdev / this  ≈ dataset Letter_Size_SD units
WORD_SPACING_SCALE = 40.0    # px gap stdev / this ≈ dataset Word_Spacing_SD units
BASELINE_DEV_SCALE = 8.0     # px baseline stdev / this ≈ dataset Baseline_Deviation units
HESITATION_THRESHOLD_SEC = 3.0  # word-quiz response slower than this counts as a hesitation


def _normalise_words(text: str) -> list[str]:
    return re.findall(r"[a-z']+", (text or "").lower())


def extract_speech_dyslexia_features(payload: dict) -> dict:
    """Build the dyslexia model's 11 inputs from an age 5–12 oral reading.

    Uses speech_service word-level alignment for precise error taxonomy and timing.
    """
    from services.speech_service import align_speech

    age = float(payload.get("age", 10))
    expected = payload.get("expected_text", "")
    spoken = payload.get("transcript", "")
    duration = max(float(payload.get("duration_sec", 0) or 0), 0.1)
    pauses = payload.get("pause_durations_sec", []) or []
    confidence = float(payload.get("speech_confidence", 0.75) or 0.75)
    accent = payload.get("accent", "en-IN")

    alignment = align_speech(
        expected_text=expected,
        recognized_transcript=spoken,
        duration_sec=duration,
        pause_durations_sec=pauses,
        speech_confidence=confidence,
        accent=accent
    )
    m = alignment["metrics"]

    return {
        "Age": age,
        "Reading_Time_sec": round(duration, 2),
        "Reading_Speed_WPM": m["reading_speed_wpm"],
        "Reading_Accuracy": m["reading_accuracy_pct"],
        "Word_Error_Count": m["total_errors"],
        "Letter_Reversal_Count": m["reversal_count"],
        "Spelling_Accuracy": m["reading_accuracy_pct"],
        "Comprehension_Score": m["reading_accuracy_pct"],
        "Avg_Response_Time_ms": m["avg_response_time_ms"],
        "Hesitation_Count": m["hesitation_count"],
        "Confidence_Score": m["speech_confidence"],
    }


def extract_dyslexia_features(payload: dict) -> dict:
    """
    payload expected shape:
    {
      "age": 11,
      "passage_word_count": 60,
      "reading_time_sec": 48.0,
      "comprehension_correct": 4, "comprehension_total": 5,
      "word_items": [{"word":"...", "correct": true/false, "response_time_sec": 2.1}, ...],
      "reversal_errors": 2, "reversal_total": 8,
      "spelling_correct": 3, "spelling_total": 5,
      "confidence_pct": 82
    }
    """
    age = float(payload.get("age", 10))
    reading_time = float(payload.get("reading_time_sec", 0) or 0)
    passage_words = float(payload.get("passage_word_count", 0) or 0)
    reading_speed_wpm = round((passage_words / (reading_time / 60)), 2) if reading_time > 0 else 0.0

    items = payload.get("word_items", [])
    n = max(len(items), 1)
    correct_count = sum(1 for it in items if it.get("correct"))
    word_error_count = n - correct_count if items else 0
    reading_accuracy = round(100 * correct_count / n, 2) if items else 100.0

    response_times = [float(it.get("response_time_sec", 0)) for it in items]
    avg_response_time_ms = round(float(np.mean(response_times)) * 1000, 1) if response_times else 0.0
    hesitation_count = sum(1 for rt in response_times if rt > HESITATION_THRESHOLD_SEC)

    reversal_errors = float(payload.get("reversal_errors", 0) or 0)

    spelling_correct = float(payload.get("spelling_correct", 0) or 0)
    spelling_total = float(payload.get("spelling_total", 1) or 1)
    spelling_accuracy = round(100 * spelling_correct / spelling_total, 2)

    comp_correct = float(payload.get("comprehension_correct", 0) or 0)
    comp_total = float(payload.get("comprehension_total", 1) or 1)
    comprehension_score = round(100 * comp_correct / comp_total)

    confidence_score = round(float(payload.get("confidence_pct", 90)) / 100, 3)

    return {
        "Age": age,
        "Reading_Time_sec": round(reading_time, 2),
        "Reading_Speed_WPM": reading_speed_wpm,
        "Reading_Accuracy": reading_accuracy,
        "Word_Error_Count": word_error_count,
        "Letter_Reversal_Count": reversal_errors,
        "Spelling_Accuracy": spelling_accuracy,
        "Comprehension_Score": comprehension_score,
        "Avg_Response_Time_ms": avg_response_time_ms,
        "Hesitation_Count": hesitation_count,
        "Confidence_Score": confidence_score,
    }


def extract_dysgraphia_features(payload: dict) -> dict:
    """Extract normalized, short-task handwriting measures from canvas strokes.

    Coordinates are normalized by canvas dimensions, tiny accidental strokes
    are discarded, motion is summarized robustly, pauses are measured between
    strokes, and spatially overlapping strokes are grouped into letter-like
    units before size/spacing/baseline consistency is estimated.
    """
    age = float(payload.get("age", 10))
    raw_strokes = payload.get("strokes", [])
    total_time = float(payload.get("total_time_sec", 0) or 0)
    letter_count = max(float(payload.get("letter_count", 0) or 0), 1)
    correction_count = int(payload.get("correction_count", 0) or 0)
    canvas_width = max(float(payload.get("canvas_width", 800) or 800), 1)
    canvas_height = max(float(payload.get("canvas_height", 340) or 340), 1)

    strokes = []
    for stroke in raw_strokes:
        if len(stroke) < 2:
            continue
        path = sum(
            (((stroke[i]["x"] - stroke[i - 1]["x"]) / canvas_width) ** 2 +
             ((stroke[i]["y"] - stroke[i - 1]["y"]) / canvas_height) ** 2) ** 0.5
            for i in range(1, len(stroke))
        )
        if path >= 0.002:  # reject taps and accidental dots
            strokes.append(stroke)

    stroke_count = len(strokes)
    pen_lift_count = max(stroke_count - 1, 0)
    writing_speed_cps = round(letter_count / total_time, 3) if total_time > 0 else 0.0

    normalized_speeds = []
    turn_angles = []
    boxes = []
    pressure_values = []
    on_surface_time = 0.0

    for stroke in strokes:
        ys = [float(p["y"]) / canvas_height for p in stroke]
        xs = [float(p["x"]) / canvas_width for p in stroke]
        boxes.append({"x0": min(xs), "x1": max(xs), "y0": min(ys), "y1": max(ys),
                      "cy": (min(ys) + max(ys)) / 2})
        on_surface_time += max(float(stroke[-1]["t"]) - float(stroke[0]["t"]), 0)
        pressure_values.extend(float(p.get("pressure", 0) or 0) for p in stroke if float(p.get("pressure", 0) or 0) > 0)
        directions = []
        for i in range(1, len(stroke)):
            dx = (float(stroke[i]["x"]) - float(stroke[i - 1]["x"])) / canvas_width
            dy = (float(stroke[i]["y"]) - float(stroke[i - 1]["y"])) / canvas_height
            dt = max(float(stroke[i]["t"]) - float(stroke[i - 1]["t"]), 0.005)
            dist = (dx ** 2 + dy ** 2) ** 0.5
            if dist > 0:
                normalized_speeds.append(dist / dt)
                directions.append(float(np.arctan2(dy, dx)))
        for a, b in zip(directions, directions[1:]):
            delta = abs((b - a + np.pi) % (2 * np.pi) - np.pi)
            turn_angles.append(delta / np.pi)

    if normalized_speeds:
        speeds = np.asarray(normalized_speeds)
        lo, hi = np.percentile(speeds, [10, 90])
        robust = np.clip(speeds, lo, hi)
        median_speed = float(np.median(robust))
        speed_irregularity = float(np.std(robust) / (np.mean(robust) + 1e-6))
        turn_irregularity = float(np.mean(turn_angles)) if turn_angles else 0.25
        stroke_smoothness = round(float(np.clip(1 - 0.38 * speed_irregularity - 0.22 * turn_irregularity, 0.30, 0.99)), 3)
    else:
        median_speed = 0.0
        speed_irregularity = 1.0
        stroke_smoothness = 0.6

    # Cluster spatially overlapping strokes within the upper/lower writing line.
    clusters = []
    for line_boxes in ([b for b in boxes if b["cy"] < 0.5], [b for b in boxes if b["cy"] >= 0.5]):
        current = None
        for box in sorted(line_boxes, key=lambda b: b["x0"]):
            if current is None or box["x0"] > current["x1"] + 0.010:
                current = dict(box)
                clusters.append(current)
            else:
                current["x0"] = min(current["x0"], box["x0"])
                current["x1"] = max(current["x1"], box["x1"])
                current["y0"] = min(current["y0"], box["y0"])
                current["y1"] = max(current["y1"], box["y1"])

    heights = [c["y1"] - c["y0"] for c in clusters if c["y1"] > c["y0"]]
    letter_size_cv = float(np.std(heights) / (np.mean(heights) + 1e-6)) if len(heights) > 2 else 0.25
    line_baselines = []
    horizontal_gaps = []
    for line in (0, 1):
        items = sorted([c for c in clusters if ((c["y0"] + c["y1"]) / 2 < 0.5) == (line == 0)], key=lambda c: c["x0"])
        if len(items) > 2:
            line_baselines.append(float(np.std([c["y1"] for c in items])))
            horizontal_gaps.extend(max(items[i]["x0"] - items[i - 1]["x1"], 0) for i in range(1, len(items)))
    baseline_deviation = float(np.mean(line_baselines)) if line_baselines else 0.025
    positive_gaps = [g for g in horizontal_gaps if g > 0.002]
    spacing_cv = float(np.std(positive_gaps) / (np.mean(positive_gaps) + 1e-6)) if len(positive_gaps) > 2 else 0.35

    ordered = sorted(strokes, key=lambda s: float(s[0]["t"]))
    pauses = [max(float(ordered[i][0]["t"]) - float(ordered[i - 1][-1]["t"]), 0) for i in range(1, len(ordered))]
    meaningful_pauses = [p for p in pauses if p >= 0.4]
    long_pauses = [p for p in pauses if p >= 1.5]
    on_surface_ratio = on_surface_time / total_time if total_time > 0 else 0
    pressure_cv = float(np.std(pressure_values) / (np.mean(pressure_values) + 1e-6)) if len(pressure_values) > 5 else 0.0

    return {
        "Age": age,
        "Writing_Time_sec": round(total_time, 2),
        "Writing_Speed_CPS": writing_speed_cps,
        "Stroke_Count": stroke_count,
        "Pen_Lift_Count": pen_lift_count,
        "Avg_Stroke_Speed": round(median_speed * 100, 3),
        "Stroke_Smoothness": stroke_smoothness,
        "Letter_Size_SD": round(letter_size_cv, 3),
        "Word_Spacing_SD": round(spacing_cv, 3),
        "Baseline_Deviation": round(baseline_deviation, 4),
        "Correction_Count": correction_count,
        "Pause_Count": len(meaningful_pauses),
        "Long_Pause_Count": len(long_pauses),
        "Mean_Pause_sec": round(float(np.mean(meaningful_pauses)), 3) if meaningful_pauses else 0.0,
        "On_Surface_Ratio": round(float(np.clip(on_surface_ratio, 0, 1)), 3),
        "Pressure_CV": round(pressure_cv, 3),
        "Estimated_Letter_Groups": len(clusters),
        "Speed_Irregularity": round(speed_irregularity, 3),
    }
