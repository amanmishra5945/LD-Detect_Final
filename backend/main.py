"""
main.py — FastAPI Backend
Smart Learning Disability Screening System (Ages 5–12)

Architecture:
Frontend <-> REST API <-> Assessment Services <-> Feature Extraction <-> ML / Scoring <-> SQLite DB
Screening support only — not a clinical diagnosis.
"""
import sys
import json
import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

# Ensure backend directory is in sys.path so imports work seamlessly
# whether executed from repo root or backend folder (critical for PaaS/Docker)
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from fastapi import FastAPI, Depends, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field

from config import (
    APP_TITLE, APP_VERSION, APP_DESCRIPTION, MIN_AGE, MAX_AGE,
    ALLOWED_ORIGINS, get_tier_label, get_age_tier
)
from database import init_db, get_db, User, AssessmentSession, Assessment
from services.speech_service import align_speech, verify_spoken_word
from services.dyslexia_service import generate_dyslexia_session, evaluate_dyslexia_session
from services.dysgraphia_service import generate_dysgraphia_session, evaluate_dysgraphia_session
from vision.handwriting_analyzer import analyze_handwriting_task
from features import (
    extract_dyslexia_features, extract_dysgraphia_features,
    extract_speech_dyslexia_features
)
from predictor import predict, predict_dysgraphia_calibrated

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    description=APP_DESCRIPTION,
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if FRONTEND_DIR.exists():
    css_dir = FRONTEND_DIR / "css"
    if css_dir.exists():
        app.mount("/css", StaticFiles(directory=str(css_dir)), name="css")
    js_dir = FRONTEND_DIR / "js"
    if js_dir.exists():
        app.mount("/js", StaticFiles(directory=str(js_dir)), name="js")


@app.get("/")
def serve_index():
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"status": "ok", "app": APP_TITLE, "version": APP_VERSION}


@app.get("/favicon.ico", include_in_schema=False)
def serve_favicon():
    svg_icon = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
        '<text y=".9em" font-size="90">🧠</text></svg>'
    )
    return Response(content=svg_icon, media_type="image/svg+xml")



# =====================================================================
# SCHEMAS
# =====================================================================

class UserCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    age: int = Field(default=8, ge=MIN_AGE, le=MAX_AGE)
    role: str = "student"


class UserUpdate(BaseModel):
    name: Optional[str] = None
    age: Optional[int] = Field(default=None, ge=MIN_AGE, le=MAX_AGE)


class SessionStartRequest(BaseModel):
    user_id: int
    age: Optional[int] = Field(default=None, ge=MIN_AGE, le=MAX_AGE)


class SpeechAlignmentRequest(BaseModel):
    expected_text: str = Field(min_length=1)
    transcript: str = Field(default="")
    duration_sec: float = Field(default=1.0, ge=0.1)
    pauses: List[float] = []
    speech_confidence: float = Field(default=0.85, ge=0.0, le=1.0)
    accent: Optional[str] = "en-IN"


class WordVerificationRequest(BaseModel):
    target_word: str = Field(min_length=1)
    spoken_transcript: str = Field(default="")
    age: int = Field(default=8, ge=MIN_AGE, le=MAX_AGE)
    response_time_sec: float = Field(default=1.0, ge=0.0)
    speech_confidence: float = Field(default=0.85, ge=0.0, le=1.0)
    accent: Optional[str] = "en-IN"



class DyslexiaSessionSubmit(BaseModel):
    session_id: Optional[int] = None
    user_id: int
    age: int = Field(ge=MIN_AGE, le=MAX_AGE)
    stages_data: Dict[str, Any]
    confidence_score: Optional[float] = 0.85
    accent: Optional[str] = "en-IN"


class TaskAnalyzeRequest(BaseModel):
    target: str
    age: int = Field(default=8, ge=MIN_AGE, le=MAX_AGE)
    strokes: List[List[Dict[str, Any]]]
    duration_sec: float = Field(default=5.0, ge=0.1)
    corrections: int = 0
    image_base64: Optional[str] = None
    canvas_width: int = 800
    canvas_height: int = 340


class DysgraphiaSessionSubmit(BaseModel):
    session_id: Optional[int] = None
    user_id: int
    age: int = Field(ge=MIN_AGE, le=MAX_AGE)
    task_results: List[Dict[str, Any]]
    device_type: Optional[str] = "external_stylus_tablet"


# --- Legacy Schemas for backward compatibility ---

class DyslexiaPayload(BaseModel):
    user_id: int
    age: int = Field(default=10, ge=MIN_AGE, le=MAX_AGE)
    passage_word_count: Optional[int] = 0
    reading_time_sec: Optional[float] = 0
    comprehension_correct: Optional[int] = 0
    comprehension_total: Optional[int] = 1
    word_items: List[Dict[str, Any]] = []
    reversal_errors: Optional[int] = 0
    reversal_total: Optional[int] = 8
    spelling_correct: Optional[int] = 0
    spelling_total: Optional[int] = 1
    confidence_pct: Optional[float] = 90


class SpeechDyslexiaPayload(BaseModel):
    user_id: int
    age: int = Field(default=10, ge=MIN_AGE, le=MAX_AGE)
    expected_text: str = Field(min_length=5, max_length=2000)
    transcript: str = Field(min_length=1, max_length=4000)
    duration_sec: float = Field(gt=0, le=900)
    speech_confidence: float = Field(default=0.75, ge=0, le=1)
    pause_durations_sec: List[float] = []
    language: str = "en-IN"


class DysgraphiaPayload(BaseModel):
    user_id: int
    age: int = Field(default=10, ge=MIN_AGE, le=MAX_AGE)
    strokes: List[List[Dict[str, Any]]]
    total_time_sec: Optional[float] = 0
    letter_count: Optional[int] = 0
    correction_count: Optional[int] = 0
    device_type: Optional[str] = "external_tablet"
    average_pressure: Optional[float] = 0
    canvas_width: float = Field(default=800, gt=0, le=5000)
    canvas_height: float = Field(default=340, gt=0, le=5000)


# =====================================================================
# GENERAL & HEALTH
# =====================================================================

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "app": APP_TITLE,
        "version": APP_VERSION,
        "age_range": f"{MIN_AGE}–{MAX_AGE}",
        "medical_disclaimer": "Educational screening aid only — not a clinical diagnosis."
    }


# =====================================================================
# USER PROFILE MANAGEMENT
# =====================================================================

@app.get("/api/users")
def list_users(db: Session = Depends(get_db)):
    users = db.query(User).order_by(User.created_at.desc()).all()
    return [
        {
            "id": u.id,
            "name": u.name,
            "age": u.age or 8,
            "role": u.role,
            "difficulty_label": get_tier_label(u.age or 8),
            "created_at": u.created_at.isoformat()
        }
        for u in users
    ]


@app.post("/api/users")
def create_user(payload: UserCreate, db: Session = Depends(get_db)):
    user = User(name=payload.name.strip(), age=payload.age, role=payload.role)
    db.add(user)
    db.commit()
    db.refresh(user)
    return {
        "id": user.id,
        "name": user.name,
        "age": user.age,
        "role": user.role,
        "difficulty_label": get_tier_label(user.age),
        "created_at": user.created_at.isoformat()
    }


@app.get("/api/users/{user_id}")
def get_user(user_id: int, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Child profile not found")
    return {
        "id": user.id,
        "name": user.name,
        "age": user.age or 8,
        "role": user.role,
        "difficulty_label": get_tier_label(user.age or 8),
        "created_at": user.created_at.isoformat()
    }


@app.put("/api/users/{user_id}")
def update_user(user_id: int, payload: UserUpdate, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Child profile not found")
    if payload.name is not None:
        user.name = payload.name.strip()
    if payload.age is not None:
        user.age = payload.age
    db.commit()
    db.refresh(user)
    return {
        "id": user.id,
        "name": user.name,
        "age": user.age or 8,
        "role": user.role,
        "difficulty_label": get_tier_label(user.age or 8)
    }


# =====================================================================
# DYSLEXIA SCREENING WORKFLOW
# =====================================================================

@app.post("/api/dyslexia/session/start")
def start_dyslexia_session(payload: SessionStartRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == payload.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Child profile not found")

    age = payload.age if payload.age is not None else (user.age or 8)
    session_data = generate_dyslexia_session(age)

    record = AssessmentSession(
        user_id=user.id,
        disorder="dyslexia",
        age=age,
        difficulty_tier=session_data["age_tier"],
        tasks_json=json.dumps(session_data["stages"])
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    return {
        "session_id": record.id,
        "user_id": user.id,
        "child_name": user.name,
        "age": age,
        "difficulty_label": session_data["difficulty_label"],
        "stages": session_data["stages"]
    }


@app.post("/api/dyslexia/align-speech")
def align_speech_endpoint(payload: SpeechAlignmentRequest, age: int = Query(default=8, ge=MIN_AGE, le=MAX_AGE)):
    """
    Performs real-time word-level alignment between expected text and recognized transcript,
    and runs real-time Random Forest classification on the live speech stream.
    """
    result = align_speech(
        expected_text=payload.expected_text,
        recognized_transcript=payload.transcript,
        duration_sec=payload.duration_sec,
        pause_durations_sec=payload.pauses,
        speech_confidence=payload.speech_confidence,
        accent=payload.accent or "en-IN"
    )
    m = result["metrics"]
    live_features = {
        "Age": float(age),
        "Reading_Time_sec": round(m["duration_sec"], 2),
        "Reading_Speed_WPM": m["reading_speed_wpm"],
        "Reading_Accuracy": m["reading_accuracy_pct"],
        "Word_Error_Count": m["total_errors"],
        "Letter_Reversal_Count": m["reversal_count"],
        "Spelling_Accuracy": m["reading_accuracy_pct"],
        "Comprehension_Score": m["reading_accuracy_pct"],
        "Avg_Response_Time_ms": m["avg_response_time_ms"],
        "Hesitation_Count": m["hesitation_count"],
        "Confidence_Score": m["speech_confidence"]
    }
    pred_res = predict("dyslexia", live_features)
    result["live_prediction"] = {
        "risk_level": pred_res["risk_level"],
        "confidence": pred_res["confidence"],
        "probabilities": pred_res["probabilities"],
        "recommendations": pred_res["recommendations"],
        "word_progress": f"{m['spoken_word_count']} of {m['expected_word_count']}",
        "features": live_features
    }
    return result


@app.post("/api/dyslexia/verify-word")
def verify_word_endpoint(payload: WordVerificationRequest):
    """
    Automated AI verification of an individual spoken word against the expected target.
    Automatically determines correctness, detects letter/word reversals, and evaluates latency.
    """
    return verify_spoken_word(
        target_word=payload.target_word,
        spoken_word=payload.spoken_transcript,
        age=payload.age,
        response_time_sec=payload.response_time_sec,
        speech_confidence=payload.speech_confidence,
        accent=payload.accent or "en-IN"
    )


@app.post("/api/dyslexia/session/submit")
def submit_dyslexia_session(payload: DyslexiaSessionSubmit, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == payload.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Child profile not found")

    result = evaluate_dyslexia_session(payload.model_dump())
    result["child_name"] = user.name
    result["age"] = payload.age

    # Save to AssessmentSession table
    session_record = None
    if payload.session_id:
        session_record = db.query(AssessmentSession).filter(AssessmentSession.id == payload.session_id).first()

    if session_record:
        session_record.prediction = result["prediction"]
        session_record.risk_level = result["risk_level"]
        session_record.confidence = result["confidence"]
        session_record.observations_json = json.dumps(result["explainable_report"]["what_we_observed"])
        session_record.features_json = json.dumps(result["features_used"])
        session_record.full_result_json = json.dumps(result)
        session_record.completed_at = datetime.datetime.now(datetime.timezone.utc)
    else:
        session_record = AssessmentSession(
            user_id=user.id,
            disorder="dyslexia",
            age=payload.age,
            difficulty_tier=get_age_tier(payload.age),
            prediction=result["prediction"],
            risk_level=result["risk_level"],
            confidence=result["confidence"],
            observations_json=json.dumps(result["explainable_report"]["what_we_observed"]),
            features_json=json.dumps(result["features_used"]),
            full_result_json=json.dumps(result),
            completed_at=datetime.datetime.now(datetime.timezone.utc)
        )
        db.add(session_record)

    # Also save to legacy Assessment table for historical compatibility
    legacy_rec = Assessment(
        user_id=user.id,
        disorder="dyslexia",
        features_json=json.dumps(result["features_used"]),
        prediction=result["prediction"],
        risk_level=result["risk_level"],
        confidence=result["confidence"]
    )
    db.add(legacy_rec)

    db.commit()
    db.refresh(session_record)
    result["session_id"] = session_record.id
    return result


# =====================================================================
# DYSGRAPHIA SCREENING WORKFLOW
# =====================================================================

@app.post("/api/dysgraphia/session/start")
def start_dysgraphia_session(payload: SessionStartRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == payload.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Child profile not found")

    age = payload.age if payload.age is not None else (user.age or 8)
    session_data = generate_dysgraphia_session(age)

    record = AssessmentSession(
        user_id=user.id,
        disorder="dysgraphia",
        age=age,
        difficulty_tier=session_data["age_tier"],
        tasks_json=json.dumps(session_data["tasks"])
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    return {
        "session_id": record.id,
        "user_id": user.id,
        "child_name": user.name,
        "age": age,
        "difficulty_label": session_data["difficulty_label"],
        "tasks": session_data["tasks"]
    }


@app.post("/api/dysgraphia/analyze-task")
def analyze_task_endpoint(payload: TaskAnalyzeRequest):
    """
    Analyzes an individual handwriting task canvas using OpenCV and stroke kinematics.
    Returns stroke smoothness, baseline drift, Hu moments, and letter mismatch flags.
    """
    res = analyze_handwriting_task(
        strokes=payload.strokes,
        target_text=payload.target,
        age=payload.age,
        total_duration_sec=payload.duration_sec,
        correction_count=payload.corrections,
        image_base64=payload.image_base64,
        canvas_width=payload.canvas_width,
        canvas_height=payload.canvas_height
    )
    return res


@app.post("/api/dysgraphia/session/submit")
def submit_dysgraphia_session(payload: DysgraphiaSessionSubmit, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == payload.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Child profile not found")

    result = evaluate_dysgraphia_session(payload.model_dump())
    result["child_name"] = user.name
    result["age"] = payload.age

    # Save to AssessmentSession table
    session_record = None
    if payload.session_id:
        session_record = db.query(AssessmentSession).filter(AssessmentSession.id == payload.session_id).first()

    if session_record:
        session_record.prediction = result["prediction"]
        session_record.risk_level = result["risk_level"]
        session_record.confidence = result["confidence"]
        session_record.screening_score = result["screening_score"]
        session_record.observations_json = json.dumps(result["explainable_report"]["what_we_observed"])
        session_record.features_json = json.dumps(result["features_used"])
        session_record.full_result_json = json.dumps(result)
        session_record.completed_at = datetime.datetime.now(datetime.timezone.utc)
    else:
        session_record = AssessmentSession(
            user_id=user.id,
            disorder="dysgraphia",
            age=payload.age,
            difficulty_tier=get_age_tier(payload.age),
            prediction=result["prediction"],
            risk_level=result["risk_level"],
            confidence=result["confidence"],
            screening_score=result["screening_score"],
            observations_json=json.dumps(result["explainable_report"]["what_we_observed"]),
            features_json=json.dumps(result["features_used"]),
            full_result_json=json.dumps(result),
            completed_at=datetime.datetime.now(datetime.timezone.utc)
        )
        db.add(session_record)

    # Legacy table record
    legacy_rec = Assessment(
        user_id=user.id,
        disorder="dysgraphia",
        features_json=json.dumps(result["features_used"]),
        prediction=result["prediction"],
        risk_level=result["risk_level"],
        confidence=result["confidence"]
    )
    db.add(legacy_rec)

    db.commit()
    db.refresh(session_record)
    result["session_id"] = session_record.id
    return result


# =====================================================================
# RESULTS & HISTORY
# =====================================================================

@app.get("/api/users/{user_id}/history")
def get_user_history(user_id: int, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Child profile not found")

    # Prefer detailed assessment sessions
    sessions = (
        db.query(AssessmentSession)
        .filter(AssessmentSession.user_id == user_id)
        .order_by(AssessmentSession.created_at.desc())
        .all()
    )

    if sessions:
        results = []
        for s in sessions:
            full_res = json.loads(s.full_result_json) if s.full_result_json else {}
            obs = json.loads(s.observations_json) if s.observations_json else []
            feats = json.loads(s.features_json) if s.features_json else {}
            results.append({
                "id": s.id,
                "session_id": s.id,
                "disorder": s.disorder,
                "prediction": s.prediction or "Normal",
                "risk_level": s.risk_level or "Normal",
                "confidence": s.confidence or 0.85,
                "screening_score": s.screening_score,
                "observations": obs,
                "features": feats,
                "full_result": full_res,
                "created_at": s.created_at.isoformat()
            })
        return results

    # Fallback to legacy assessments table if no sessions exist yet
    legacy_recs = (
        db.query(Assessment)
        .filter(Assessment.user_id == user_id)
        .order_by(Assessment.created_at.desc())
        .all()
    )
    return [
        {
            "id": r.id,
            "session_id": r.id,
            "disorder": r.disorder,
            "prediction": r.prediction,
            "risk_level": r.risk_level,
            "confidence": r.confidence,
            "screening_score": 0,
            "observations": [],
            "features": json.loads(r.features_json) if r.features_json else {},
            "full_result": {},
            "created_at": r.created_at.isoformat(),
        }
        for r in legacy_recs
    ]


# =====================================================================
# BACKWARD COMPATIBLE LEGACY ROUTES
# =====================================================================

@app.post("/api/dyslexia/speech-assess")
def assess_dyslexia_from_speech(payload: SpeechDyslexiaPayload, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == payload.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Child profile not found")

    features = extract_speech_dyslexia_features(payload.model_dump())
    result = predict("dyslexia", features)
    result["assessment_mode"] = "speech_recognition"
    result["transcript"] = payload.transcript
    result["language"] = payload.language

    record = Assessment(
        user_id=user.id, disorder="dyslexia", features_json=json.dumps(features),
        prediction=result["prediction"], risk_level=result["risk_level"],
        confidence=result["confidence"],
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    result["assessment_id"] = record.id
    return result


@app.post("/api/dyslexia/assess")
def assess_dyslexia(payload: DyslexiaPayload, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == payload.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Child profile not found")

    features = extract_dyslexia_features(payload.model_dump())
    result = predict("dyslexia", features)

    record = Assessment(
        user_id=user.id,
        disorder="dyslexia",
        features_json=json.dumps(features),
        prediction=result["prediction"],
        risk_level=result["risk_level"],
        confidence=result["confidence"],
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    result["assessment_id"] = record.id
    return result


@app.post("/api/dysgraphia/assess")
def assess_dysgraphia(payload: DysgraphiaPayload, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == payload.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Child profile not found")

    features = extract_dysgraphia_features(payload.model_dump())
    if features["Stroke_Count"] < 4 or features["Writing_Time_sec"] < 3:
        raise HTTPException(
            status_code=422,
            detail="Writing sample is too brief. Please complete the writing task before submitting.",
        )
    result = predict_dysgraphia_calibrated(features)
    result["assessment_mode"] = "external_tablet_handwriting"
    result["input_device"] = payload.device_type
    result["sample_quality"] = "standard" if payload.device_type == "external_stylus_tablet" else "demo_pointer"

    record = Assessment(
        user_id=user.id,
        disorder="dysgraphia",
        features_json=json.dumps(features),
        prediction=result["prediction"],
        risk_level=result["risk_level"],
        confidence=result["confidence"],
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    result["assessment_id"] = record.id
    return result
