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


def test_dysgraphia_target_matching_and_scribble_detection():
    import base64
    import cv2
    import numpy as np

    # Create test child profile
    u_res = client.post("/api/users", json={"name": "Handwriting Test Child", "age": 8})
    uid = u_res.json()["id"]

    def make_b64(draw_fn):
        img = np.zeros((340, 800), dtype=np.uint8)
        draw_fn(img)
        _, buf = cv2.imencode(".png", img)
        return "data:image/png;base64," + base64.b64encode(buf).decode("utf-8")

    # 1. Test Scribbles / straight lines: Must NOT be Normal
    def draw_scribble(img):
        cv2.line(img, (100, 170), (700, 170), 255, 3)

    scribble_payload = {
        "user_id": uid,
        "age": 8,
        "device_type": "stylus",
        "task_results": [
            {"target": "b", "strokes": [], "duration_sec": 3.0, "corrections": 0, "image_base64": make_b64(draw_scribble), "canvas_width": 800, "canvas_height": 340},
            {"target": "d", "strokes": [], "duration_sec": 3.0, "corrections": 0, "image_base64": make_b64(draw_scribble), "canvas_width": 800, "canvas_height": 340},
            {"target": "cat", "strokes": [], "duration_sec": 5.0, "corrections": 0, "image_base64": make_b64(draw_scribble), "canvas_width": 800, "canvas_height": 340},
            {"target": "sun", "strokes": [], "duration_sec": 5.0, "corrections": 0, "image_base64": make_b64(draw_scribble), "canvas_width": 800, "canvas_height": 340},
            {"target": "red ball", "strokes": [], "duration_sec": 8.0, "corrections": 0, "image_base64": make_b64(draw_scribble), "canvas_width": 800, "canvas_height": 340},
        ]
    }
    res_sc = client.post("/api/dysgraphia/session/submit", json=scribble_payload)
    assert res_sc.status_code == 200
    d_sc = res_sc.json()
    assert d_sc["prediction"] in ("Moderate", "Severe"), f"Expected Moderate/Severe for scribbles, got {d_sc['prediction']}"
    assert d_sc["matched_tasks_count"] == 0
    assert d_sc["unmatched_tasks_count"] == 5
    assert d_sc["score_breakdown"]["content_formation_accuracy"] >= 30

    # 2. Test Accurate Writing: Must be Normal
    def draw_text(txt):
        def fn(img):
            cv2.putText(img, txt, (80, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.8 if len(txt)<=2 else 1.2, 255, 3, cv2.LINE_AA)
        return fn

    clean_payload = {
        "user_id": uid,
        "age": 8,
        "device_type": "stylus",
        "task_results": [
            {"target": "b", "strokes": [], "duration_sec": 3.0, "corrections": 0, "image_base64": make_b64(draw_text("b")), "canvas_width": 800, "canvas_height": 340},
            {"target": "d", "strokes": [], "duration_sec": 3.0, "corrections": 0, "image_base64": make_b64(draw_text("d")), "canvas_width": 800, "canvas_height": 340},
            {"target": "cat", "strokes": [], "duration_sec": 5.0, "corrections": 0, "image_base64": make_b64(draw_text("cat")), "canvas_width": 800, "canvas_height": 340},
            {"target": "sun", "strokes": [], "duration_sec": 5.0, "corrections": 0, "image_base64": make_b64(draw_text("sun")), "canvas_width": 800, "canvas_height": 340},
            {"target": "red ball", "strokes": [], "duration_sec": 8.0, "corrections": 0, "image_base64": make_b64(draw_text("red ball")), "canvas_width": 800, "canvas_height": 340},
        ]
    }
    res_cl = client.post("/api/dysgraphia/session/submit", json=clean_payload)
    assert res_cl.status_code == 200
    d_cl = res_cl.json()
    assert d_cl["prediction"] == "Normal"
    assert d_cl["screening_score"] < 20
    assert d_cl["matched_tasks_count"] == 5

    # 3. Test b/d letter reversal
    rev_payload = {
        "user_id": uid,
        "age": 8,
        "device_type": "stylus",
        "task_results": [
            {"target": "b", "strokes": [], "duration_sec": 3.0, "corrections": 0, "image_base64": make_b64(draw_text("d")), "canvas_width": 800, "canvas_height": 340},
        ]
    }
    res_rv = client.post("/api/dysgraphia/session/submit", json=rev_payload)
    assert res_rv.status_code == 200
    d_rv = res_rv.json()
    assert d_rv["score_breakdown"]["letter_form_and_reversals"] > 0
    obs_text = " ".join(d_rv["explainable_report"]["what_we_observed"])
    assert "reversal" in obs_text.lower()


def test_dysgraphia_transparent_rgba_and_natural_writing():
    import io
    import base64
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw

    u_res = client.post("/api/users", json={"name": "Canvas Fidelity Child", "age": 7})
    uid = u_res.json()["id"]

    def make_transparent_rgba_b64(draw_fn):
        # Simulate browser canvas: RGBA transparent (0,0,0,0) with #0F172A ink
        img = Image.new("RGBA", (760, 320), (0, 0, 0, 0))
        draw_fn(img)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

    def draw_transparent_b(img):
        draw = ImageDraw.Draw(img)
        # Vertical stem
        draw.line([(100, 80), (100, 240)], fill=(15, 23, 42, 255), width=4)
        # Lower loop on right side
        draw.arc([(100, 160), (180, 240)], start=270, end=90, fill=(15, 23, 42, 255), width=4)

    def draw_transparent_spaced_cat(img):
        # Natural handwriting where letters c, a, t have space between them
        arr = np.array(img)
        cv2.putText(arr, "c", (100, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (15, 23, 42, 255), 3, cv2.LINE_AA)
        cv2.putText(arr, "a", (220, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (15, 23, 42, 255), 3, cv2.LINE_AA)
        cv2.putText(arr, "t", (340, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (15, 23, 42, 255), 3, cv2.LINE_AA)
        return Image.fromarray(arr)

    def make_arr_b64(arr_img):
        buf = io.BytesIO()
        arr_img.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

    # Spaced cat image in transparent RGBA
    empty_rgba = Image.new("RGBA", (760, 320), (0, 0, 0, 0))
    cat_img = draw_transparent_spaced_cat(empty_rgba)
    b64_cat = make_arr_b64(cat_img)

    # Uppercase SUN in transparent RGBA
    empty_rgba2 = Image.new("RGBA", (760, 320), (0, 0, 0, 0))
    arr2 = np.array(empty_rgba2)
    cv2.putText(arr2, "SUN", (120, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (15, 23, 42, 255), 3, cv2.LINE_AA)
    sun_img = Image.fromarray(arr2)
    b64_sun = make_arr_b64(sun_img)

    # Stroke coordinates test (without image_base64)
    stroke_cat = [
        [{"x": 100, "y": 150, "t": 0.0}, {"x": 80, "y": 180, "t": 0.2}, {"x": 120, "y": 200, "t": 0.4}],
        [{"x": 150, "y": 170, "t": 0.6}, {"x": 170, "y": 170, "t": 0.8}],
        [{"x": 200, "y": 140, "t": 1.0}, {"x": 200, "y": 200, "t": 1.2}],
    ]

    payload = {
        "user_id": uid,
        "age": 7,
        "device_type": "stylus",
        "task_results": [
            {"target": "b", "strokes": [], "duration_sec": 3.0, "corrections": 0, "image_base64": make_transparent_rgba_b64(draw_transparent_b), "canvas_width": 760, "canvas_height": 320},
            {"target": "cat", "strokes": [], "duration_sec": 4.0, "corrections": 0, "image_base64": b64_cat, "canvas_width": 760, "canvas_height": 320},
            {"target": "sun", "strokes": [], "duration_sec": 4.0, "corrections": 0, "image_base64": b64_sun, "canvas_width": 760, "canvas_height": 320},
            {"target": "tree", "strokes": stroke_cat, "duration_sec": 4.0, "corrections": 0, "image_base64": None, "canvas_width": 760, "canvas_height": 320},
        ]
    }

    res = client.post("/api/dysgraphia/session/submit", json=payload)
    assert res.status_code == 200
    data = res.json()

    # The transparent canvas images must NOT be reported as "blank or insufficient ink"
    for task_res in data["per_task_results"]:
        ver = task_res["verification"]
        for r in ver.get("reasons", []):
            assert "insufficient ink" not in r.lower()

    # Targets 'b', 'cat', and 'sun' must be matched successfully
    res_b = data["per_task_results"][0]["verification"]
    assert res_b["is_matched"] is True, f"b was not matched: {res_b}"

    res_cat = data["per_task_results"][1]["verification"]
    assert res_cat["is_matched"] is True, f"Spaced cat was not matched: {res_cat}"

    res_sun = data["per_task_results"][2]["verification"]
    assert res_sun["is_matched"] is True, f"Uppercase SUN was not matched: {res_sun}"


def test_dysgraphia_severity_thresholds():
    from services.dysgraphia_service import evaluate_dysgraphia_session

    u_res = client.post("/api/users", json={"name": "Thresholds Child", "age": 8})
    uid = u_res.json()["id"]

    def make_mock_payload(mock_accuracy):
        # Generate dummy task results that return the specified mock_accuracy
        import cv2, numpy as np, base64
        img = np.zeros((340, 800), dtype=np.uint8)
        if mock_accuracy >= 60:
            cv2.putText(img, "cat", (80, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 255, 3)
        elif mock_accuracy < 40:
            cv2.line(img, (100, 170), (700, 170), 255, 3)
        _, buf = cv2.imencode(".png", img)
        b64 = "data:image/png;base64," + base64.b64encode(buf).decode("utf-8")
        return {
            "session_id": None,
            "user_id": uid,
            "age": 8,
            "device_type": "stylus",
            "task_results": [
                {"target": "cat", "strokes": [], "duration_sec": 4.0, "corrections": 0, "image_base64": b64, "canvas_width": 800, "canvas_height": 340}
            ]
        }

    # Severe case (< 40%)
    res_sev = client.post("/api/dysgraphia/session/submit", json=make_mock_payload(15))
    d_sev = res_sev.json()
    assert d_sev["target_accuracy_pct"] < 40.0
    assert d_sev["prediction"] == "Severe"

    # Normal case (> 60%)
    res_norm = client.post("/api/dysgraphia/session/submit", json=make_mock_payload(90))
    d_norm = res_norm.json()
    assert d_norm["target_accuracy_pct"] >= 60.0
    assert d_norm["prediction"] == "Normal"

    # Direct validation of all four brackets via evaluate_dysgraphia_session
    from unittest.mock import patch
    cases = [
        (70.0, "Normal"),
        (60.0, "Normal"),
        (55.0, "Mild"),
        (50.0, "Mild"),
        (45.0, "Moderate"),
        (40.0, "Moderate"),
        (35.0, "Severe"),
        (10.0, "Severe"),
    ]
    for acc, expected_label in cases:
        mock_ev = {
            "target": "cat",
            "metrics": {
                "stroke_count": 3, "pen_lifts": 2, "duration_sec": 4.0, "writing_speed_cps": 0.8,
                "stroke_smoothness": 0.8, "speed_irregularity": 0.2, "median_speed": 20.0,
                "letter_size_cv": 0.3, "spacing_cv": 0.3, "baseline_deviation": 0.02,
                "meaningful_pauses": 0, "long_pauses": 0, "mean_pause_sec": 0.0,
                "ink_density": 0.01, "num_contours": 3, "corrections": 0
            },
            "cv_features": {"fill_density": 0.01, "contour_count": 3, "hu_moments": [0.0]*4},
            "visual_similarity": acc / 100.0,
            "verification": {"is_matched": acc >= 60.0, "accuracy_pct": acc, "has_reversal": False, "reasons": []},
            "letter_form_flag": None,
            "char_classification": None,
            "observations": []
        }
        with patch("services.dysgraphia_service.analyze_handwriting_task", return_value=mock_ev):
            res_eval = evaluate_dysgraphia_session({"user_id": uid, "age": 8, "task_results": [{"target": "cat"}]})
            assert res_eval["prediction"] == expected_label, f"For {acc}%, expected {expected_label} but got {res_eval['prediction']}"




