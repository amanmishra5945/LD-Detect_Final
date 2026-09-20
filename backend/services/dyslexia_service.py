"""
services/dyslexia_service.py
----------------------------
Orchestrates the 7-stage Dyslexia screening workflow:
Stage 1: Letter Recognition
Stage 2: Word Recognition
Stage 3: Oral Reading (word-by-word microphone alignment)
Stage 4: Phonological Awareness
Stage 5: Spelling Analysis
Stage 6: Reading Comprehension
Stage 7: Rapid Naming (RAN)

Integrates the validated Random Forest classifier with an explainable multi-evidence
scoring rubric. Guaranteed strictly non-clinical screening terminology.
"""
import json
import random
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

from config import get_age_tier, get_tier_label, DATA_DIR
from services.speech_service import align_speech
from predictor import predict, RECOMMENDATIONS

TEST_BANK_PATH = DATA_DIR / "dyslexia_test_bank.json"


def load_test_bank() -> Dict[str, Any]:
    with open(TEST_BANK_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_dyslexia_session(age: int) -> Dict[str, Any]:
    """
    Generates an age-tailored set of tasks across all 7 stages.
    Guarantees unique items without repetition within the session.
    """
    bank = load_test_bank()
    tier = get_age_tier(age)
    
    # 1. Letter recognition (pick 4-6 letters including confusable pairs)
    letters_pool = bank.get("letter_recognition", {}).get(tier, [])
    sample_letters = random.sample(letters_pool, min(5, len(letters_pool)))
    
    # 2. Word recognition (pick 5 words)
    words_pool = bank.get("word_recognition", {}).get(tier, [])
    sample_words = random.sample(words_pool, min(5, len(words_pool)))
    
    # 3. Oral reading passage (pick 1 sentence/passage)
    passages_pool = bank.get("oral_reading", {}).get(tier, [])
    sample_passage = random.choice(passages_pool) if passages_pool else {"text": "The dog runs fast."}
    
    # 4. Phonological awareness (pick 3 tasks)
    phono_pool = bank.get("phonological_awareness", {}).get(tier, [])
    sample_phono = random.sample(phono_pool, min(3, len(phono_pool)))
    
    # 5. Spelling (pick 3 words)
    spelling_pool = bank.get("spelling", {}).get(tier, [])
    sample_spelling = random.sample(spelling_pool, min(3, len(spelling_pool)))
    
    # 6. Comprehension (pick 1 passage with question)
    comp_pool = bank.get("comprehension", {}).get(tier, [])
    sample_comp = random.choice(comp_pool) if comp_pool else None
    
    # 7. Rapid naming (pick 6 items)
    rn_pool = bank.get("rapid_naming", {}).get(tier, [])
    sample_rn = random.sample(rn_pool, min(6, len(rn_pool)))
    
    return {
        "age": age,
        "age_tier": tier,
        "difficulty_label": get_tier_label(age),
        "stages": [
            {
                "stage": 1,
                "name": "Letter Recognition",
                "instruction": "Read each letter aloud clearly.",
                "items": sample_letters
            },
            {
                "stage": 2,
                "name": "Word Recognition",
                "instruction": "Read each word aloud one at a time.",
                "items": [{"word": w} for w in sample_words]
            },
            {
                "stage": 3,
                "name": "Oral Reading",
                "instruction": "Click the microphone and read the sentence aloud.",
                "passage": sample_passage
            },
            {
                "stage": 4,
                "name": "Phonological Awareness",
                "instruction": "Listen or read the question and select the correct answer.",
                "items": sample_phono
            },
            {
                "stage": 5,
                "name": "Spelling",
                "instruction": "Type the word from memory after reading the hint.",
                "items": sample_spelling
            },
            {
                "stage": 6,
                "name": "Reading Comprehension",
                "instruction": "Read the short text and answer the question.",
                "item": sample_comp
            },
            {
                "stage": 7,
                "name": "Rapid Naming",
                "instruction": "Name each item as quickly as you can.",
                "items": sample_rn
            }
        ]
    }


def analyze_spelling_error(target: str, typed: str) -> Dict[str, Any]:
    """Analyzes spelling errors at letter level: missing, extra, transpositions."""
    target = (target or "").strip().lower()
    typed = (typed or "").strip().lower()
    
    is_correct = (target == typed)
    if is_correct:
        return {"correct": True, "error_type": "none"}
        
    # Check transposition (anagram)
    if sorted(target) == sorted(typed) and len(target) == len(typed):
        return {"correct": False, "error_type": "transposition", "detail": "Letters transposed"}
        
    missing = [ch for ch in target if target.count(ch) > typed.count(ch)]
    extra = [ch for ch in typed if typed.count(ch) > target.count(ch)]
    
    return {
        "correct": False,
        "error_type": "phonetic_or_omission",
        "missing_letters": missing,
        "extra_letters": extra
    }


def evaluate_dyslexia_session(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluates a completed multi-stage session.
    Extracts the 11 features for the Random Forest model and creates an explainable
    evidence layer.
    """
    age = int(payload.get("age", 8))
    accent = payload.get("accent") or payload.get("stages_data", {}).get("stage_3_speech", {}).get("accent", "en-IN")
    stages_data = payload.get("stages_data", {})
    
    # 1. Letter Recognition
    letter_items = stages_data.get("stage_1_letters", [])
    n_letters = max(len(letter_items), 1)
    correct_letters = sum(1 for it in letter_items if it.get("correct", False))
    letter_acc = round(100.0 * correct_letters / n_letters, 2)
    letter_reversals = sum(1 for it in letter_items if it.get("reversal_error", False))
    
    # 2. Word Recognition
    word_items = stages_data.get("stage_2_words", [])
    n_words = max(len(word_items), 1)
    correct_words = sum(1 for it in word_items if it.get("correct", False))
    word_acc = round(100.0 * correct_words / n_words, 2)
    word_errors = n_words - correct_words
    word_response_times = [float(it.get("response_time_sec", 1.5)) for it in word_items]
    
    # 3. Oral Reading (Speech alignment)
    speech_data = stages_data.get("stage_3_speech", {})
    expected_passage = speech_data.get("expected_text", "The dog runs fast.")
    recognized_transcript = speech_data.get("transcript", "")
    duration_sec = max(float(speech_data.get("duration_sec", 5.0) or 5.0), 0.5)
    pauses = speech_data.get("pauses", [])
    
    alignment_result = align_speech(
        expected_text=expected_passage,
        recognized_transcript=recognized_transcript,
        duration_sec=duration_sec,
        pause_durations_sec=pauses,
        speech_confidence=float(speech_data.get("confidence", 0.85)),
        accent=accent
    )
    speech_metrics = alignment_result["metrics"]
    
    # 4. Phonological Awareness
    phono_items = stages_data.get("stage_4_phono", [])
    n_phono = max(len(phono_items), 1)
    correct_phono = sum(1 for it in phono_items if it.get("correct", False))
    phono_acc = round(100.0 * correct_phono / n_phono, 2)
    
    # 5. Spelling
    spelling_items = stages_data.get("stage_5_spelling", [])
    n_spelling = max(len(spelling_items), 1)
    correct_spelling = sum(1 for it in spelling_items if it.get("correct", False))
    spelling_acc = round(100.0 * correct_spelling / n_spelling, 2)
    
    # 6. Reading Comprehension
    comp_data = stages_data.get("stage_6_comp", {})
    comp_score = 100 if comp_data.get("correct", False) else 0
    
    # 7. Rapid Naming
    rn_data = stages_data.get("stage_7_rapid_naming", {})
    rn_duration = float(rn_data.get("total_time_sec", 8.0) or 8.0)
    rn_errors = int(rn_data.get("errors", 0))

    # Overall Reading metrics synthesis
    overall_reading_time = duration_sec + sum(word_response_times)
    total_passage_words = speech_metrics["expected_word_count"]
    reading_speed_wpm = speech_metrics["reading_speed_wpm"] if speech_metrics["reading_speed_wpm"] > 0 else round(total_passage_words / (duration_sec / 60.0), 2)
    reading_acc = round(0.5 * speech_metrics["reading_accuracy_pct"] + 0.3 * word_acc + 0.2 * letter_acc, 2)
    total_word_errors = word_errors + speech_metrics["substitutions"] + speech_metrics["omissions"]
    
    total_reversals = letter_reversals + speech_metrics["reversal_count"]
    
    all_response_times = word_response_times + [speech_metrics["avg_response_time_ms"] / 1000.0]
    avg_response_ms = round(float(np.mean(all_response_times)) * 1000.0, 1)
    
    hesitations = speech_metrics["hesitation_count"] + sum(1 for rt in word_response_times if rt > 3.0)
    confidence_score = round(float(payload.get("confidence_score", speech_metrics["speech_confidence"])), 3)

    # 11 features for Random Forest Model
    rf_features = {
        "Age": float(age),
        "Reading_Time_sec": round(overall_reading_time, 2),
        "Reading_Speed_WPM": reading_speed_wpm,
        "Reading_Accuracy": reading_acc,
        "Word_Error_Count": total_word_errors,
        "Letter_Reversal_Count": total_reversals,
        "Spelling_Accuracy": spelling_acc,
        "Comprehension_Score": comp_score,
        "Avg_Response_Time_ms": avg_response_ms,
        "Hesitation_Count": hesitations,
        "Confidence_Score": confidence_score
    }

    # Run validated Random Forest classifier
    rf_result = predict("dyslexia", rf_features)

    # Compile explainable observation layer
    observations = []
    observations.extend(alignment_result["observations"])
    
    if letter_acc < 80:
        observations.append(f"Letter recognition showed hesitation or confusion ({letter_acc}% accuracy).")
    if letter_reversals > 0:
        observations.append(f"{letter_reversals} confusable letter choice(s) noted (e.g. b/d, p/q).")
    if word_acc < 80:
        observations.append(f"Isolated word decoding accuracy was {word_acc}%.")
    if phono_acc < 70:
        observations.append(f"Phonological awareness tasks indicated difficulty with sound segmentation/rhyming ({phono_acc}%).")
    if spelling_acc < 70:
        observations.append(f"Spelling recall was below typical range ({spelling_acc}%).")
    if rn_duration > 15.0:
        observations.append("Rapid naming speed was notably slow, which may correlate with lexical retrieval speed.")

    if not observations:
        observations.append("All observed reading and speech indicators were within age-appropriate ranges.")

    what_this_means = (
        "The assessment analyzed phonological decoding, word recognition, oral reading fluency, "
        "letter-sound correspondence, and rapid retrieval. "
    )
    if rf_result["prediction"] in ("Moderate", "Severe"):
        what_this_means += (
            "Observed patterns suggest several reading fluency and decoding difficulties that "
            "may benefit from structured phonics guidance."
        )
    elif rf_result["prediction"] == "Mild":
        what_this_means += (
            "Minor variations in reading speed, spelling, or letter recognition were observed. "
            "These may reflect early developmental variations or emerging skills."
        )
    else:
        what_this_means += "Performance showed age-appropriate reading fluency and phonological processing."

    return {
        "disorder": "dyslexia",
        "accent": accent,
        "prediction": rf_result["prediction"],
        "risk_level": rf_result["risk_level"],
        "confidence": rf_result["confidence"],
        "probabilities": rf_result["probabilities"],
        "screening_title": "Dyslexia Screening Result",
        "disclaimer": "Screening support only — not a clinical diagnosis.",
        "stage_breakdown": {
            "letter_recognition_accuracy": letter_acc,
            "word_recognition_accuracy": word_acc,
            "oral_reading_accuracy": speech_metrics["reading_accuracy_pct"],
            "reading_speed_wpm": reading_speed_wpm,
            "phonological_accuracy": phono_acc,
            "spelling_accuracy": spelling_acc,
            "comprehension_accuracy": comp_score,
            "rapid_naming_sec": rn_duration,
            "hesitation_count": hesitations,
            "letter_reversals": total_reversals
        },
        "speech_alignment": alignment_result,
        "features_used": rf_features,
        "explainable_report": {
            "what_we_observed": observations,
            "what_this_means": what_this_means,
            "suggested_next_steps": rf_result["recommendations"]
        }
    }
