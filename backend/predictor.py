"""
predictor.py
Prediction Engine — loads the trained multi-class Random Forest models,
scalers, and label encoders, and turns a feature dict into a full
classification result: predicted class, per-class probabilities, confidence,
and mapped recommendations. Matches the "Prediction Engine" box in the
Architecture diagram, upgraded to genuine multi-class output.
"""
import os
import joblib
import numpy as np

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

_cache = {}


def _load(name):
    if name not in _cache:
        clf = joblib.load(os.path.join(MODELS_DIR, f"{name}_rf.joblib"))
        scaler = joblib.load(os.path.join(MODELS_DIR, f"{name}_scaler.joblib"))
        feature_cols = joblib.load(os.path.join(MODELS_DIR, f"{name}_features.joblib"))
        le = joblib.load(os.path.join(MODELS_DIR, f"{name}_label_encoder.joblib"))
        _cache[name] = (clf, scaler, feature_cols, le)
    return _cache[name]


RECOMMENDATIONS = {
    "dyslexia": {
        "Severe": [
            "Recommend a full clinical evaluation with a learning specialist / psychologist as a priority",
            "Intensive multi-sensory phonics program (e.g., Orton-Gillingham style) with a trained tutor",
            "Assistive tech: text-to-speech for all reading material, extended time on assessments",
        ],
        "Moderate": [
            "Structured, guided phonics and phonemic-awareness sessions, 3-4x per week",
            "Text-to-speech assisted reading practice for grade-level material",
            "Re-screen in 4-6 weeks; consider a professional evaluation if no improvement",
        ],
        "Mild": [
            "Extra guided reading practice, 2-3x per week, with a focus on flagged error patterns",
            "Letter-reversal and phonics reinforcement games",
            "Re-screen in 6-8 weeks to confirm trend",
        ],
        "Normal": [
            "Continue regular grade-level reading practice",
            "Periodic re-screening (once per term) as a precaution",
        ],
    },
    "dysgraphia": {
        "Severe": [
            "Recommend a full clinical evaluation with an occupational therapist as a priority",
            "Structured OT program: tactile letter formation, grip and posture correction",
            "Assistive tech: allow keyboarding/typing for written assignments where possible",
        ],
        "Moderate": [
            "Regular fine-motor and handwriting drills, 3-4x per week",
            "Use pencil grips, wide-ruled or raised-line paper",
            "Re-screen in 4-6 weeks; consider a professional OT evaluation if no improvement",
        ],
        "Mild": [
            "Short daily handwriting drills focused on letter consistency and spacing",
            "Encourage slower, deliberate writing over speed",
            "Re-screen in 6-8 weeks to confirm trend",
        ],
        "Normal": [
            "Continue regular handwriting practice",
            "Periodic re-screening (once per term) as a precaution",
        ],
    },
}

# Used by the frontend to color/sort risk bands consistently
SEVERITY_ORDER = ["Normal", "Mild", "Moderate", "Severe"]


def predict(disorder: str, feature_dict: dict) -> dict:
    clf, scaler, feature_cols, le = _load(disorder)
    x = np.array([[feature_dict[c] for c in feature_cols]])
    x_scaled = scaler.transform(x)

    pred_idx = int(clf.predict(x_scaled)[0])
    proba = clf.predict_proba(x_scaled)[0]  # aligned to clf.classes_ (encoded ints)
    predicted_label = le.inverse_transform([pred_idx])[0]
    confidence = float(proba[list(clf.classes_).index(pred_idx)])

    # Full probability breakdown by class name, in a stable Normal->Severe order
    class_names_by_encoded = {i: name for i, name in enumerate(le.classes_)}
    prob_by_class = {}
    for encoded_class, p in zip(clf.classes_, proba):
        prob_by_class[class_names_by_encoded[encoded_class]] = round(float(p), 4)
    prob_breakdown = {c: prob_by_class.get(c, 0.0) for c in SEVERITY_ORDER}

    recs = RECOMMENDATIONS[disorder][predicted_label]

    return {
        "disorder": disorder,
        "prediction": predicted_label,          # Normal / Mild / Moderate / Severe
        "risk_level": predicted_label,           # kept for frontend badge compatibility
        "confidence": round(confidence, 4),
        "probabilities": prob_breakdown,
        "features_used": feature_dict,
        "recommendations": recs,
    }


def predict_dysgraphia_calibrated(feature_dict: dict, vision_results: dict = None) -> dict:
    """Explainable two-sentence screening for the age 5–10 canvas.

    This intentionally does not call the legacy Random Forest, whose training
    samples were much longer than the two-sentence task. Scores are bounded and
    every contribution is returned for display and audit.

    Args:
        feature_dict: Extracted kinematic/spatial features from canvas strokes.
        vision_results: Optional dict from classify_canvas_characters() with
                        keys like 'reversal_ratio', 'corrected_ratio',
                        'vision_risk_score', 'summary', etc.
    """
    age = int(round(float(feature_dict["Age"])))
    speed = float(feature_dict["Writing_Speed_CPS"])
    smoothness = float(feature_dict["Stroke_Smoothness"])
    corrections = int(feature_dict["Correction_Count"])
    letter_size_cv = float(feature_dict["Letter_Size_SD"])
    spacing_cv = float(feature_dict["Word_Spacing_SD"])
    baseline = float(feature_dict["Baseline_Deviation"])
    mean_pause = float(feature_dict.get("Mean_Pause_sec", 0))
    long_pauses = int(feature_dict.get("Long_Pause_Count", 0))
    on_surface_ratio = float(feature_dict.get("On_Surface_Ratio", 0))
    pressure_cv = float(feature_dict.get("Pressure_CV", 0))
    estimated_groups = int(feature_dict.get("Estimated_Letter_Groups", 0))

    # Broad, deliberately conservative age bands for a short copying task.
    if age <= 6:
        speed_low, speed_high = 0.25, 1.20
    elif age <= 8:
        speed_low, speed_high = 0.35, 1.60
    elif age <= 10:
        speed_low, speed_high = 0.50, 2.20
    else:
        speed_low, speed_high = 0.70, 2.80

    components = {}
    reasons = []

    if speed < speed_low * 0.5:
        components["writing_speed"] = 20
        reasons.append("Writing speed was far below the broad age-adjusted range")
    elif speed < speed_low or speed > speed_high * 1.6:
        components["writing_speed"] = 10
        reasons.append("Writing speed was outside the broad age-adjusted range")
    else:
        components["writing_speed"] = 0

    if smoothness >= 0.78:
        components["stroke_smoothness"] = 0
    elif smoothness >= 0.62:
        components["stroke_smoothness"] = 8
        reasons.append("Some variation was detected in stroke smoothness")
    elif smoothness >= 0.48:
        components["stroke_smoothness"] = 16
        reasons.append("Stroke movement was noticeably inconsistent")
    else:
        components["stroke_smoothness"] = 25
        reasons.append("Stroke movement was highly inconsistent")

    components["corrections"] = min(corrections * 5, 10)
    if corrections:
        reasons.append(f"{corrections} correction(s) were recorded")

    pause_score = 0
    if mean_pause > 1.8:
        pause_score = 12
    elif mean_pause > 1.2:
        pause_score = 8
    elif mean_pause > 0.8:
        pause_score = 4
    pause_score += min(long_pauses * 2, 5)
    if on_surface_ratio and on_surface_ratio < 0.25:
        pause_score += 3
    components["pauses_and_hesitation"] = min(pause_score, 15)
    if components["pauses_and_hesitation"]:
        reasons.append("Long or frequent pauses reduced handwriting fluency")

    consistency = 0
    if letter_size_cv > 0.80:
        consistency += 8
    elif letter_size_cv > 0.55:
        consistency += 5
    elif letter_size_cv > 0.35:
        consistency += 2
    if spacing_cv > 1.20:
        consistency += 7
    elif spacing_cv > 0.85:
        consistency += 4
    if baseline > 0.10:
        consistency += 8
    elif baseline > 0.06:
        consistency += 5
    elif baseline > 0.035:
        consistency += 2
    if estimated_groups and estimated_groups < 8:
        consistency += 3
    components["visual_consistency"] = min(consistency, 25)
    if consistency:
        reasons.append("Letter size, spacing or baseline showed inconsistency")

    if pressure_cv == 0:
        components["pressure_stability"] = 0
    elif pressure_cv <= 0.35:
        components["pressure_stability"] = 0
    elif pressure_cv <= 0.55:
        components["pressure_stability"] = 2
        reasons.append("Pen pressure varied mildly")
    else:
        components["pressure_stability"] = 5
        reasons.append("Pen pressure varied considerably")

    # Vision-based character classification (max 15 pts)
    if vision_results and vision_results.get("model_available"):
        rev_ratio = float(vision_results.get("reversal_ratio", 0))
        cor_ratio = float(vision_results.get("corrected_ratio", 0))
        total_chars = int(vision_results.get("total_characters", 0))
        summary = vision_results.get("summary", {})
        v_score = 0
        if rev_ratio > 0.5:
            v_score += 12
            reasons.append(
                f"Vision model detected reversed letter-forms in "
                f"{summary.get('Reversal', 0)}/{total_chars} characters"
            )
        elif rev_ratio > 0.2:
            v_score += 7
            reasons.append(
                f"Vision model detected some reversed letter-forms "
                f"({summary.get('Reversal', 0)}/{total_chars} characters)"
            )
        elif rev_ratio > 0:
            v_score += 3
        if cor_ratio > 0.3:
            v_score += 3
            reasons.append(
                f"Vision model detected self-correction patterns in "
                f"{summary.get('Corrected', 0)}/{total_chars} characters"
            )
        components["vision_classification"] = min(v_score, 15)
    else:
        components["vision_classification"] = 0

    score = int(min(sum(components.values()), 100))
    if score < 20:
        label = "Normal"
    elif score < 40:
        label = "Mild"
    elif score < 65:
        label = "Moderate"
    else:
        label = "Severe"

    centers = {"Normal": 8, "Mild": 29, "Moderate": 52, "Severe": 78}
    weights = {name: float(np.exp(-abs(score - center) / 10)) for name, center in centers.items()}
    total = sum(weights.values())
    probabilities = {name: round(weights[name] / total, 4) for name in SEVERITY_ORDER}
    confidence = probabilities[label]

    if not reasons:
        reasons.append("Captured writing measures were within the broad screening ranges")

    return {
        "disorder": "dysgraphia",
        "prediction": label,
        "risk_level": label,
        "confidence": round(confidence, 4),
        "probabilities": probabilities,
        "features_used": feature_dict,
        "recommendations": RECOMMENDATIONS["dysgraphia"][label],
        "screening_score": score,
        "score_breakdown": components,
        "assessment_reasons": reasons,
        "method": "normalized_two_sentence_rubric_v3",
    }

