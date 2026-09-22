"""
services/dysgraphia_service.py
------------------------------
Orchestrates the 5-task child-friendly Dysgraphia screening workflow:
Task 1: Write a letter (e.g. 'A' or 'd')
Task 2: Write another letter (e.g. 'b' or 'p')
Task 3: Write a simple word (e.g. 'cat' or 'tree')
Task 4: Write another word (e.g. 'sun' or 'school')
Task 5: Copy an age-appropriate phrase or sentence

Combines OpenCV computer vision analysis with stroke kinematics.
Produces explainable multi-component scores and non-clinical parent reports.
"""
import json
import random
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

from config import get_age_tier, get_tier_label, DATA_DIR
from vision.handwriting_analyzer import analyze_handwriting_task
from predictor import RECOMMENDATIONS, SEVERITY_ORDER

TEST_BANK_PATH = DATA_DIR / "dysgraphia_test_bank.json"


def load_test_bank() -> Dict[str, Any]:
    with open(TEST_BANK_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_dysgraphia_session(age: int) -> Dict[str, Any]:
    """
    Generates an age-tailored 5-task handwriting assessment.
    Selects balanced tasks from the test bank with alternating content.
    """
    bank = load_test_bank()
    tier = get_age_tier(age)
    
    alternates = bank.get("alternates", {}).get(tier, {})
    default_tasks = bank.get("tasks", {}).get(tier, [])
    
    # Create 5 randomized tasks based on age tier
    letters = alternates.get("letters", ["b", "d"])
    words = alternates.get("words", ["cat", "sun"])
    phrases = alternates.get("phrases", ["red ball"]) if "phrases" in alternates else alternates.get("sentences", ["The sun is warm."])
    
    selected_letters = random.sample(letters, min(2, len(letters))) if len(letters) >= 2 else ["b", "d"]
    selected_words = random.sample(words, min(2, len(words))) if len(words) >= 2 else ["cat", "sun"]
    selected_phrase = random.choice(phrases) if phrases else "The sun is warm."
    
    tasks = [
        {
            "task_index": 1,
            "type": "letter",
            "target": selected_letters[0],
            "instruction": f"Write the letter: {selected_letters[0]}"
        },
        {
            "task_index": 2,
            "type": "letter",
            "target": selected_letters[1],
            "instruction": f"Write the letter: {selected_letters[1]}"
        },
        {
            "task_index": 3,
            "type": "word",
            "target": selected_words[0],
            "instruction": f"Write the word: {selected_words[0]}"
        },
        {
            "task_index": 4,
            "type": "word",
            "target": selected_words[1],
            "instruction": f"Write the word: {selected_words[1]}"
        },
        {
            "task_index": 5,
            "type": "phrase",
            "target": selected_phrase,
            "instruction": f"Copy this: {selected_phrase}"
        }
    ]
    
    return {
        "age": age,
        "age_tier": tier,
        "difficulty_label": get_tier_label(age),
        "tasks": tasks
    }


def evaluate_dysgraphia_session(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluates all completed handwriting tasks from a session.
    Runs Computer Vision and kinematic analysis on each task canvas,
    then synthesizes a calibrated, explainable multi-component score.
    """
    age = int(payload.get("age", 8))
    task_results = payload.get("task_results", [])
    device_type = payload.get("device_type", "external_stylus_tablet")

    if not task_results:
        # Fallback for empty submission
        return {
            "disorder": "dysgraphia",
            "prediction": "Normal",
            "risk_level": "Normal",
            "confidence": 0.85,
            "screening_title": "Dysgraphia Screening Result",
            "screening_score": 10,
            "observations": ["No writing samples provided."]
        }

    per_task_evals = []
    for tr in task_results:
        target = tr.get("target", "")
        strokes = tr.get("strokes", [])
        duration = float(tr.get("duration_sec", 5.0) or 5.0)
        corrections = int(tr.get("corrections", 0))
        img_b64 = tr.get("image_base64", None)
        c_w = int(tr.get("canvas_width", 800) or 800)
        c_h = int(tr.get("canvas_height", 340) or 340)

        ev = analyze_handwriting_task(
            strokes=strokes,
            target_text=target,
            age=age,
            total_duration_sec=duration,
            correction_count=corrections,
            image_base64=img_b64,
            canvas_width=c_w,
            canvas_height=c_h
        )
        per_task_evals.append(ev)

    # Aggregate kinematic & CV features across tasks
    total_strokes = sum(e["metrics"]["stroke_count"] for e in per_task_evals)
    total_pen_lifts = sum(e["metrics"]["pen_lifts"] for e in per_task_evals)
    total_duration = sum(e["metrics"]["duration_sec"] for e in per_task_evals)
    total_corrections = sum(e["metrics"]["corrections"] for e in per_task_evals)
    
    avg_speed_cps = float(np.mean([e["metrics"]["writing_speed_cps"] for e in per_task_evals]))
    avg_smoothness = float(np.mean([e["metrics"]["stroke_smoothness"] for e in per_task_evals]))
    avg_letter_size_cv = float(np.mean([e["metrics"]["letter_size_cv"] for e in per_task_evals]))
    avg_spacing_cv = float(np.mean([e["metrics"]["spacing_cv"] for e in per_task_evals]))
    avg_baseline_dev = float(np.mean([e["metrics"]["baseline_deviation"] for e in per_task_evals]))
    avg_visual_similarity = float(np.mean([e["visual_similarity"] for e in per_task_evals]))
    total_long_pauses = sum(e["metrics"]["long_pauses"] for e in per_task_evals)

    letter_form_flags = [e["letter_form_flag"] for e in per_task_evals if e.get("letter_form_flag")]

    # Aggregate vision-based character classification results
    vision_reversals = 0
    vision_corrected = 0
    vision_total_chars = 0
    for ev in per_task_evals:
        cc = ev.get("char_classification")
        if cc and cc.get("model_available"):
            s = cc.get("summary", {})
            vision_reversals += s.get("Reversal", 0)
            vision_corrected += s.get("Corrected", 0)
            vision_total_chars += cc.get("total_characters", 0)

    # Target Word & Letter Fidelity Verification
    task_verifications = [ev.get("verification", {}) for ev in per_task_evals]
    unmatched_tasks = [ev for ev in per_task_evals if not ev.get("verification", {}).get("is_matched", True)]
    unmatched_count = len(unmatched_tasks)
    total_tasks = max(len(per_task_evals), 1)
    target_accuracies = [ev.get("verification", {}).get("accuracy_pct", 70) for ev in per_task_evals]
    avg_target_accuracy = float(np.mean(target_accuracies)) if target_accuracies else 80.0

    reversal_tasks = [
        ev for ev in per_task_evals
        if ev.get("verification", {}).get("has_reversal", False) or ev.get("letter_form_flag")
    ]
    reversal_count = len(reversal_tasks)

    # Calibrated Explainable Scoring (0–100 scale; 0 = perfect fluency/control, >60 = higher indicator load)
    # Age expected speed ranges (CPS)
    if age <= 6:
        exp_speed_min, exp_speed_max = 0.20, 1.20
    elif age <= 8:
        exp_speed_min, exp_speed_max = 0.35, 1.60
    elif age <= 10:
        exp_speed_min, exp_speed_max = 0.50, 2.20
    else:
        exp_speed_min, exp_speed_max = 0.70, 2.80

    components = {}
    reasons = []

    # 1. Target Content Formation & Accuracy (max 35 pts)
    # Inability to write the requested letters/words or drawing scribbles is a primary indicator
    if unmatched_count >= 4:
        content_score = 35
    elif unmatched_count == 3:
        content_score = 27
    elif unmatched_count == 2:
        content_score = 18
    elif unmatched_count == 1:
        content_score = 10
    else:
        content_score = 0

    if avg_target_accuracy < 50.0 and content_score < 35:
        content_score = min(content_score + 6, 35)

    components["content_formation_accuracy"] = content_score

    if unmatched_count > 0:
        reasons.append(
            f"Content Fidelity: {unmatched_count} of {total_tasks} written task(s) did not match the "
            f"requested target characters or words (average match: {avg_target_accuracy:.0f}%)."
        )
        for ev in unmatched_tasks[:3]:
            r_list = ev.get("verification", {}).get("reasons", [])
            detail = r_list[0] if r_list else f"Drawn shape did not match target '{ev.get('target', '')}'."
            reasons.append(f"Task '{ev.get('target', '')}': {detail}")

    # 2. Letter-Form Fidelity & Reversals (max 25 pts)
    visual_score = 0
    if reversal_count >= 2:
        visual_score += 22
        reasons.append(f"Multiple letter reversals detected ({reversal_count} tasks showed inverted letter orientations).")
    elif reversal_count == 1:
        visual_score += 14
        rev_target = reversal_tasks[0].get("target", "")
        reasons.append(f"Letter reversal detected on '{rev_target}' (e.g. confused orientation b/d or p/q).")

    if letter_form_flags and visual_score < 20:
        for flag in letter_form_flags[:2]:
            if flag not in [r for r in reasons]:
                reasons.append(f"Observed indicator: {flag}")

    if avg_visual_similarity < 0.50 and visual_score < 25:
        visual_score = min(visual_score + 8, 25)
        reasons.append("Visual structural similarity to the target characters was below typical threshold.")

    components["letter_form_and_reversals"] = min(visual_score, 25)

    # 3. Motor Control & Smoothness (max 20 pts)
    if avg_smoothness >= 0.78:
        components["motor_control"] = 0
    elif avg_smoothness >= 0.62:
        components["motor_control"] = 6
        reasons.append("Mild tremor or trajectory irregularity detected in stroke movement.")
    elif avg_smoothness >= 0.48:
        components["motor_control"] = 13
        reasons.append("Stroke movement exhibited noticeable motor inconsistency.")
    else:
        components["motor_control"] = 20
        reasons.append("High stroke irregularity and erratic pen trajectory observed.")

    # 4. Spatial Organization & Consistency (max 20 pts)
    spatial_score = 0
    if avg_letter_size_cv > 0.70:
        spatial_score += 7
        reasons.append("Character height and letter sizes varied inconsistently.")
    elif avg_letter_size_cv > 0.45:
        spatial_score += 4

    if avg_spacing_cv > 1.10:
        spatial_score += 7
        reasons.append("Inter-letter and word spacing was notably uneven.")
    elif avg_spacing_cv > 0.80:
        spatial_score += 4

    if avg_baseline_dev > 0.08:
        spatial_score += 8
        reasons.append("Written text showed marked deviation from a straight horizontal baseline.")
    elif avg_baseline_dev > 0.045:
        spatial_score += 4
    components["spatial_organization"] = min(spatial_score, 20)

    # 5. Writing Fluency & Speed (max 15 pts)
    if avg_speed_cps < exp_speed_min * 0.5:
        components["writing_fluency"] = 15
        reasons.append(f"Writing speed ({avg_speed_cps:.2f} char/sec) was substantially below the expected age range.")
    elif avg_speed_cps < exp_speed_min or avg_speed_cps > exp_speed_max * 1.6:
        components["writing_fluency"] = 8
        reasons.append(f"Writing speed ({avg_speed_cps:.2f} char/sec) was outside the typical age range.")
    else:
        components["writing_fluency"] = 0

    # 6. Pauses and Hesitations (max 10 pts)
    pause_score = min(total_long_pauses * 3, 10)
    components["pauses_and_hesitations"] = pause_score
    if pause_score >= 6:
        reasons.append(f"Frequent pauses ({total_long_pauses} hesitation pauses >1.5s) interrupted writing fluency.")

    # 7. Corrections contribution (max 5 pts)
    if total_corrections > 0:
        components["corrections"] = min(total_corrections * 2, 5)
        reasons.append(f"{total_corrections} stroke correction(s)/undo(s) were recorded.")
    else:
        components["corrections"] = 0

    # 8. Vision-based character classification (max 10 pts)
    if vision_total_chars > 0:
        rev_ratio = vision_reversals / vision_total_chars
        cor_ratio = vision_corrected / vision_total_chars
        vision_score = 0
        if rev_ratio > 0.5:
            vision_score += 8
            reasons.append(
                f"Vision model detected reversed letter-forms in "
                f"{vision_reversals}/{vision_total_chars} characters."
            )
        elif rev_ratio > 0.2:
            vision_score += 5
            reasons.append(
                f"Vision model detected some reversed letter-forms "
                f"({vision_reversals}/{vision_total_chars} characters)."
            )
        elif rev_ratio > 0:
            vision_score += 2

        if cor_ratio > 0.3:
            vision_score += 2
            reasons.append(
                f"Vision model detected self-correction patterns in "
                f"{vision_corrected}/{vision_total_chars} characters."
            )
        components["vision_classification"] = min(vision_score, 10)
    else:
        components["vision_classification"] = 0

    # Classification directly calibrated by written text match fidelity:
    # >= 60%: Normal
    # 50% - 60%: Mild
    # 40% - 50%: Moderate
    # < 40%: Severe
    if avg_target_accuracy >= 60.0:
        predicted_label = "Normal"
        screening_score = int(np.clip(100.0 - avg_target_accuracy, 0, 19))
    elif avg_target_accuracy >= 50.0:
        predicted_label = "Mild"
        screening_score = int(np.clip(20.0 + (60.0 - avg_target_accuracy) * 2.0, 20, 39))
    elif avg_target_accuracy >= 40.0:
        predicted_label = "Moderate"
        screening_score = int(np.clip(42.0 + (50.0 - avg_target_accuracy) * 2.5, 42, 67))
    else:
        predicted_label = "Severe"
        screening_score = int(np.clip(68.0 + (40.0 - avg_target_accuracy) * 0.8, 68, 100))

    centers = {"Normal": 85.0, "Mild": 55.0, "Moderate": 45.0, "Severe": 25.0}
    weights = {name: float(np.exp(-abs(avg_target_accuracy - center) / 10.0)) for name, center in centers.items()}
    w_sum = sum(weights.values())
    probabilities = {name: round(weights[name] / w_sum, 4) for name in SEVERITY_ORDER}
    confidence = probabilities[predicted_label]

    if not reasons:
        reasons.append("All handwriting fluency, stroke smoothness, and spatial layout measures were within age-appropriate ranges.")

    what_this_means = (
        "The dysgraphia assessment evaluated fine motor control, handwriting fluency, spatial organization "
        "(size, spacing, baseline straightness), and letter-form formation across multiple writing tasks. "
    )
    if predicted_label in ("Moderate", "Severe"):
        what_this_means += (
            "Observed indicators reflect difficulties in fine-motor coordination, stroke consistency, or spatial alignment "
            "that may warrant supportive practice and further professional evaluation."
        )
    elif predicted_label == "Mild":
        what_this_means += (
            "Minor fine-motor or spatial irregularities were noted, typical of developing writing skills. "
            "Gentle guided practice is suggested."
        )
    else:
        what_this_means += "Writing fluency and spatial organization reflect typical developmental expectations."

    # Legacy 11 feature representation for compatibility
    legacy_features = {
        "Age": float(age),
        "Writing_Time_sec": round(total_duration, 2),
        "Writing_Speed_CPS": round(avg_speed_cps, 3),
        "Stroke_Count": total_strokes,
        "Pen_Lift_Count": total_pen_lifts,
        "Avg_Stroke_Speed": round(float(np.mean([e["metrics"]["median_speed"] for e in per_task_evals])), 2),
        "Stroke_Smoothness": round(avg_smoothness, 3),
        "Letter_Size_SD": round(avg_letter_size_cv, 3),
        "Word_Spacing_SD": round(avg_spacing_cv, 3),
        "Baseline_Deviation": round(avg_baseline_dev, 4),
        "Correction_Count": total_corrections
    }

    return {
        "disorder": "dysgraphia",
        "prediction": predicted_label,
        "risk_level": predicted_label,
        "confidence": confidence,
        "probabilities": probabilities,
        "screening_title": "Dysgraphia Screening Result",
        "screening_score": screening_score,
        "target_accuracy_pct": round(avg_target_accuracy, 1),
        "matched_tasks_count": total_tasks - unmatched_count,
        "unmatched_tasks_count": unmatched_count,
        "total_tasks_count": total_tasks,
        "score_breakdown": components,
        "disclaimer": "Screening support only — not a clinical diagnosis.",
        "input_device": device_type,
        "per_task_results": per_task_evals,
        "features_used": legacy_features,
        "explainable_report": {
            "what_we_observed": reasons,
            "what_this_means": what_this_means,
            "suggested_next_steps": RECOMMENDATIONS["dysgraphia"][predicted_label]
        }
    }
