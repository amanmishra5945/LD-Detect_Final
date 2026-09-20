"""
tests/test_backend.py
Comprehensive automated test suite for Smart Learning Disability Screening System (Ages 5–12).
Tests API endpoints, session lifecycles, speech alignment algorithm, computer vision handwriting analysis,
age bounds, and database audit logs.
"""
import sys
from pathlib import Path

# Add backend directory to sys.path
BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

import pytest
from fastapi.testclient import TestClient
from main import app
from database import init_db

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_db():
    init_db()


def test_health():
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "5–12" in data["age_range"]
    assert "screening" in data["medical_disclaimer"].lower()


def test_user_lifecycle_and_age_bounds():
    # 1. Valid user creation (Age 6, beginner)
    res1 = client.post("/api/users", json={"name": "Aarav Sharma", "age": 6, "role": "student"})
    assert res1.status_code == 200
    u1 = res1.json()
    assert u1["name"] == "Aarav Sharma"
    assert u1["age"] == 6
    assert "Beginner" in u1["difficulty_label"]

    # 2. Valid user creation (Age 11, advanced)
    res2 = client.post("/api/users", json={"name": "Priya Patel", "age": 11, "role": "student"})
    assert res2.status_code == 200
    u2 = res2.json()
    assert u2["age"] == 11
    assert "Advanced" in u2["difficulty_label"]

    # 3. Invalid age bounds (under 5 and over 12)
    res_low = client.post("/api/users", json={"name": "Too Young", "age": 4})
    assert res_low.status_code == 422

    res_high = client.post("/api/users", json={"name": "Too Old", "age": 13})
    assert res_high.status_code == 422

    # 4. Get user
    res_get = client.get(f"/api/users/{u1['id']}")
    assert res_get.status_code == 200
    assert res_get.json()["id"] == u1["id"]

    # 5. Update user
    res_put = client.put(f"/api/users/{u1['id']}", json={"age": 7})
    assert res_put.status_code == 200
    assert res_put.json()["age"] == 7
    assert "Elementary" in res_put.json()["difficulty_label"]


def test_speech_word_alignment_and_error_taxonomy():
    # Expected: "The little boy runs to school"
    # Recognized: "The liddle boy jumps to to school" (substitution: little->liddle, substitution: runs->jumps, repetition: to)
    payload = {
        "expected_text": "The little boy runs to school",
        "transcript": "The liddle boy jumps to to school",
        "duration_sec": 4.5,
        "pauses": [1.6, 0.4],
        "speech_confidence": 0.88
    }
    res = client.post("/api/dyslexia/align-speech", json=payload)
    assert res.status_code == 200
    data = res.json()
    
    assert "aligned_words" in data
    assert "metrics" in data
    m = data["metrics"]
    assert m["expected_word_count"] == 6
    assert m["substitutions"] >= 1
    assert m["hesitation_count"] == 1  # pause 1.6s >= 1.5s
    assert m["reading_accuracy_pct"] > 0
    assert m["reading_speed_wpm"] > 0
    assert len(data["observations"]) > 0


def test_speech_reversal_detection():
    payload = {
        "expected_text": "The dog saw the cat on the mat",
        "transcript": "The dog was the cat no the mat",
        "duration_sec": 5.0
    }
    res = client.post("/api/dyslexia/align-speech", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["metrics"]["reversal_count"] >= 1
    assert any("inversion" in obs.lower() or "confusion" in obs.lower() for obs in data["observations"])


def test_dyslexia_session_flow():
    # 1. Create child
    u_res = client.post("/api/users", json={"name": "Kiran Rao", "age": 8})
    user_id = u_res.json()["id"]

    # 2. Start session
    start_res = client.post("/api/dyslexia/session/start", json={"user_id": user_id, "age": 8})
    assert start_res.status_code == 200
    s_data = start_res.json()
    assert "session_id" in s_data
    assert len(s_data["stages"]) == 7
    assert s_data["stages"][0]["name"] == "Letter Recognition"
    assert s_data["stages"][1]["name"] == "Word Recognition"
    assert s_data["stages"][2]["name"] == "Oral Reading"

    # 3. Submit full session
    submit_payload = {
        "session_id": s_data["session_id"],
        "user_id": user_id,
        "age": 8,
        "stages_data": {
            "stage_1_letters": [
                {"target": "b", "correct": True, "reversal_error": False},
                {"target": "d", "correct": False, "reversal_error": True},
                {"target": "p", "correct": True, "reversal_error": False},
                {"target": "m", "correct": True, "reversal_error": False}
            ],
            "stage_2_words": [
                {"word": "tree", "correct": True, "response_time_sec": 1.2},
                {"word": "bird", "correct": True, "response_time_sec": 1.5},
                {"word": "jump", "correct": False, "response_time_sec": 3.4}
            ],
            "stage_3_speech": {
                "expected_text": "Two brown rabbits hop quietly through the green grass.",
                "transcript": "Two brown rabbits hop quickly through the grass.",
                "duration_sec": 6.2,
                "pauses": [1.8],
                "confidence": 0.82
            },
            "stage_4_phono": [
                {"correct": True},
                {"correct": True},
                {"correct": False}
            ],
            "stage_5_spelling": [
                {"target": "tree", "correct": True},
                {"target": "bird", "correct": False}
            ],
            "stage_6_comp": {"correct": True},
            "stage_7_rapid_naming": {"total_time_sec": 9.5, "errors": 1}
        },
        "confidence_score": 0.85
    }
    sub_res = client.post("/api/dyslexia/session/submit", json=submit_payload)
    assert sub_res.status_code == 200
    res_data = sub_res.json()
    assert res_data["disorder"] == "dyslexia"
    assert res_data["prediction"] in ("Normal", "Mild", "Moderate", "Severe")
    assert "probabilities" in res_data
    assert "explainable_report" in res_data
    assert "what_we_observed" in res_data["explainable_report"]
    assert "Screening support only" in res_data["disclaimer"]


def test_handwriting_cv_task_analysis():
    # Simulate drawing the letter 'b' with strokes
    # Vertical line down + rounded loop
    strokes = [
        [{"x": 100, "y": 50, "t": 0.0}, {"x": 100, "y": 150, "t": 0.5}],  # vertical stem
        [
            {"x": 100, "y": 100, "t": 0.6},
            {"x": 140, "y": 100, "t": 0.8},
            {"x": 140, "y": 150, "t": 1.0},
            {"x": 100, "y": 150, "t": 1.2}
        ]  # loop on right
    ]
    payload = {
        "target": "b",
        "age": 7,
        "strokes": strokes,
        "duration_sec": 1.5,
        "corrections": 0,
        "canvas_width": 800,
        "canvas_height": 340
    }
    res = client.post("/api/dysgraphia/analyze-task", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "metrics" in data
    assert "cv_features" in data
    assert data["metrics"]["stroke_count"] >= 1
    assert data["cv_features"]["fill_density"] > 0
    assert len(data["cv_features"]["hu_moments"]) >= 1
    assert data["visual_similarity"] > 0


def test_dysgraphia_session_flow():
    # 1. Create child
    u_res = client.post("/api/users", json={"name": "Dev Joshi", "age": 9})
    user_id = u_res.json()["id"]

    # 2. Start session
    start_res = client.post("/api/dysgraphia/session/start", json={"user_id": user_id, "age": 9})
    assert start_res.status_code == 200
    s_data = start_res.json()
    assert "session_id" in s_data
    assert len(s_data["tasks"]) == 5

    # 3. Submit session
    strokes_sample = [
        [{"x": 50, "y": 50, "t": 0.0}, {"x": 80, "y": 50, "t": 0.3}, {"x": 80, "y": 90, "t": 0.6}]
    ]
    sub_payload = {
        "session_id": s_data["session_id"],
        "user_id": user_id,
        "age": 9,
        "device_type": "external_stylus_tablet",
        "task_results": [
            {"target": "q", "strokes": strokes_sample, "duration_sec": 3.0, "corrections": 0},
            {"target": "m", "strokes": strokes_sample, "duration_sec": 3.5, "corrections": 0},
            {"target": "school", "strokes": strokes_sample, "duration_sec": 6.0, "corrections": 1},
            {"target": "friend", "strokes": strokes_sample, "duration_sec": 5.5, "corrections": 0},
            {"target": "The sun rises.", "strokes": strokes_sample, "duration_sec": 8.0, "corrections": 0}
        ]
    }
    sub_res = client.post("/api/dysgraphia/session/submit", json=sub_payload)
    assert sub_res.status_code == 200
    data = sub_res.json()
    assert data["disorder"] == "dysgraphia"
    assert data["prediction"] in ("Normal", "Mild", "Moderate", "Severe")
    assert 0 <= data["screening_score"] <= 100
    assert "score_breakdown" in data
    assert "explainable_report" in data
    assert "Screening support only" in data["disclaimer"]


def test_user_history_persistence():
    # Create user and verify history retrieves recent assessments
    u_res = client.post("/api/users", json={"name": "History Test", "age": 8})
    uid = u_res.json()["id"]

    hist_empty = client.get(f"/api/users/{uid}/history")
    assert hist_empty.status_code == 200
    assert len(hist_empty.json()) == 0

    # Submit a quick legacy assessment to test backward compatibility
    client.post("/api/dyslexia/assess", json={
        "user_id": uid,
        "age": 8,
        "passage_word_count": 30,
        "reading_time_sec": 25.0,
        "comprehension_correct": 1,
        "comprehension_total": 1,
        "confidence_pct": 85
    })

    hist_full = client.get(f"/api/users/{uid}/history")
    assert hist_full.status_code == 200
    assert len(hist_full.json()) >= 1
    first_rec = hist_full.json()[0]
    assert first_rec["disorder"] == "dyslexia"
    assert first_rec["prediction"] in ("Normal", "Mild", "Moderate", "Severe")


def test_automated_word_verification():
    # 1. Test exact correct word
    res_cor = client.post("/api/dyslexia/verify-word", json={
        "target_word": "dog",
        "spoken_transcript": "dog",
        "age": 8,
        "response_time_sec": 0.9,
        "speech_confidence": 0.95
    })
    assert res_cor.status_code == 200
    d_cor = res_cor.json()
    assert d_cor["is_correct"] is True
    assert d_cor["status"] == "correct"
    assert d_cor["reversal_detected"] is False

    # 2. Test letter reversal confusion (b/d)
    res_rev = client.post("/api/dyslexia/verify-word", json={
        "target_word": "dog",
        "spoken_transcript": "bog",
        "age": 8,
        "response_time_sec": 1.6,
        "speech_confidence": 0.85
    })
    assert res_rev.status_code == 200
    d_rev = res_rev.json()
    assert d_rev["is_correct"] is False
    assert d_rev["reversal_detected"] is True
    assert "reversal" in d_rev["status"]

    # 3. Test word inversion (was / saw)
    res_inv = client.post("/api/dyslexia/verify-word", json={
        "target_word": "was",
        "spoken_transcript": "saw",
        "age": 7,
        "response_time_sec": 1.2
    })
    assert res_inv.status_code == 200
    d_inv = res_inv.json()
    assert d_inv["is_correct"] is False
    assert d_inv["reversal_detected"] is True

    # 4. Test distinct substitution (cat vs dog)
    res_sub = client.post("/api/dyslexia/verify-word", json={
        "target_word": "dog",
        "spoken_transcript": "cat",
        "age": 8,
        "response_time_sec": 1.0
    })
    assert res_sub.status_code == 200
    d_sub = res_sub.json()
    assert d_sub["is_correct"] is False
    assert d_sub["status"] == "substitution"


def test_indian_accent_speech_alignment():
    # Test oral reading with natural Indian English phonetic variations
    # Target: "The cat runs very fast"
    # Recognized in Indian English: "De kat ranz wery faast"
    payload = {
        "expected_text": "The cat runs very fast",
        "transcript": "De kat ranz wery faast",
        "duration_sec": 4.0,
        "pauses": [0.3],
        "speech_confidence": 0.88,
        "accent": "en-IN"
    }
    res = client.post("/api/dyslexia/align-speech", json=payload)
    assert res.status_code == 200
    data = res.json()
    m = data["metrics"]
    assert m["expected_word_count"] == 5
    assert m["correct_words"] == 5
    assert m["substitutions"] == 0
    assert m["reading_accuracy_pct"] == 100.0
    assert m["accent_matches"] >= 3
    assert data["live_prediction"]["risk_level"] == "Normal"


def test_indian_accent_word_verification():
    # 1. Indian English dental stop: "the" -> "de"
    res_the = client.post("/api/dyslexia/verify-word", json={
        "target_word": "the",
        "spoken_transcript": "de",
        "age": 8,
        "accent": "en-IN"
    })
    assert res_the.status_code == 200
    d_the = res_the.json()
    assert d_the["is_correct"] is True
    assert d_the["accent_match"] is True
    assert d_the["reversal_detected"] is False

    # 2. V/W merger: "went" -> "vent"
    res_vent = client.post("/api/dyslexia/verify-word", json={
        "target_word": "went",
        "spoken_transcript": "vent",
        "age": 8,
        "accent": "en-IN"
    })
    assert res_vent.status_code == 200
    d_vent = res_vent.json()
    assert d_vent["is_correct"] is True
    assert d_vent["accent_match"] is True

    # 3. Dental stop in "three" -> "tree"
    res_tree = client.post("/api/dyslexia/verify-word", json={
        "target_word": "three",
        "spoken_transcript": "tree",
        "age": 8,
        "accent": "en-IN"
    })
    assert res_tree.status_code == 200
    assert res_tree.json()["is_correct"] is True

    # 4. Critical safety check: true reversals (bad vs dad) must NEVER be excused as accent matches
    res_rev = client.post("/api/dyslexia/verify-word", json={
        "target_word": "bad",
        "spoken_transcript": "dad",
        "age": 8,
        "accent": "en-IN"
    })
    assert res_rev.status_code == 200
    d_rev = res_rev.json()
    assert d_rev["is_correct"] is False
    assert d_rev["reversal_detected"] is True


def test_handwriting_classifier_and_vision_pipeline():
    from vision.handwriting_classifier import is_model_available, classify_character, classify_canvas_characters
    import numpy as np

    assert is_model_available() is True

    # Test single character classification with synthetic test image
    dummy_char = np.zeros((32, 32), dtype=np.uint8)
    dummy_char[8:24, 15:17] = 255  # Vertical stroke
    res = classify_character(dummy_char)
    assert res["model_available"] is True
    assert res["label"] in ["Normal", "Corrected", "Reversal"]
    assert "confidence" in res
    assert "probabilities" in res

    # Test full canvas character segmentation and classification
    canvas = np.zeros((100, 200), dtype=np.uint8)
    canvas[20:60, 20:30] = 255  # Stroke 1
    canvas[20:60, 60:70] = 255  # Stroke 2
    canvas_res = classify_canvas_characters(canvas)
    assert canvas_res["model_available"] is True
    assert canvas_res["total_characters"] >= 2
    assert "vision_risk_score" in canvas_res
    assert "summary" in canvas_res
