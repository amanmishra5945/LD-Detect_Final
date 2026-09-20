"""
database.py
SQLAlchemy models + SQLite engine.
Organizes Child Profile (User), AssessmentSession, and legacy Assessment tables
for reliable audit trails, repeat assessments, and historical progress.
"""
import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, text
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from config import DATABASE_URL

connect_args = {"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc)


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    age = Column(Integer, default=8)  # Age 5-12
    role = Column(String, default="student")  # student, parent, teacher
    created_at = Column(DateTime, default=utc_now)

    assessments = relationship("Assessment", back_populates="user", cascade="all, delete-orphan")
    sessions = relationship("AssessmentSession", back_populates="user", cascade="all, delete-orphan")


class AssessmentSession(Base):
    __tablename__ = "assessment_sessions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    disorder = Column(String, nullable=False)  # "dyslexia" or "dysgraphia"
    age = Column(Integer, default=8)
    difficulty_tier = Column(String, default="elementary")
    prediction = Column(String)  # "Normal" / "Mild" / "Moderate" / "Severe"
    risk_level = Column(String)  # Same as prediction for UI badge compatibility
    confidence = Column(Float, default=0.0)
    screening_score = Column(Integer, default=0)
    tasks_json = Column(String, default="[]")
    observations_json = Column(String, default="[]")
    features_json = Column(String, default="{}")
    full_result_json = Column(String, default="{}")
    created_at = Column(DateTime, default=utc_now)
    completed_at = Column(DateTime, nullable=True)

    user = relationship("User", back_populates="sessions")


class Assessment(Base):
    """Retained for backward compatibility with existing tests and history records."""
    __tablename__ = "assessments"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    disorder = Column(String)  # "dyslexia" or "dysgraphia"
    features_json = Column(String)  # extracted features, stored as JSON string
    prediction = Column(String)  # "Normal" / "Mild" / "Moderate" / "Severe"
    risk_level = Column(String)
    confidence = Column(Float)
    created_at = Column(DateTime, default=utc_now)

    user = relationship("User", back_populates="assessments")


def init_db():
    Base.metadata.create_all(bind=engine)
    # Check if 'age' column exists on 'users' table in existing sqlite db
    with engine.connect() as conn:
        try:
            res = conn.execute(text("PRAGMA table_info(users)"))
            cols = [row[1] for row in res.fetchall()]
            if "age" not in cols:
                conn.execute(text("ALTER TABLE users ADD COLUMN age INTEGER DEFAULT 8"))
                conn.commit()
        except Exception:
            pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
