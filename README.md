# Smart Learning Disability Screening System (Ages 5–12)

An intelligent, multi-stage, AI-powered screening application designed to assist educators, parents, and clinicians in identifying early observable indicators of **Dyslexia** and **Dysgraphia** in children aged **5 to 12**.

> [!IMPORTANT]
> **Ethical & Clinical Disclaimer**  
> This system is an **educational screening support tool**, **NOT a clinical diagnostic instrument**.  
> All assessments produce objective screening indicators, confidence metrics, and explainable observations to support educators and parents. Results should always be discussed with a qualified learning specialist, educational psychologist, or occupational therapist for comprehensive clinical evaluation.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Key Features](#key-features)
3. [Age-Adaptive Architecture (Ages 5–12)](#age-adaptive-architecture-ages-512)
4. [Dyslexia 7-Stage Screening Workflow](#dyslexia-7-stage-screening-workflow)
5. [Dysgraphia 5-Task Handwriting Workflow](#dysgraphia-5-task-handwriting-workflow)
6. [Speech Recognition & Alignment Engine](#speech-recognition--alignment-engine)
7. [Computer Vision & Stroke Kinematics](#computer-vision--stroke-kinematics)
8. [Machine Learning & Scoring Pipeline](#machine-learning--scoring-pipeline)
9. [System Architecture & Database](#system-architecture--database)
10. [REST API Documentation](#rest-api-documentation)
11. [Installation & Windows Setup Guide](#installation--windows-setup-guide)
12. [Presentation & Demonstration Workflow](#presentation--demonstration-workflow)
13. [Testing & Quality Assurance](#testing--quality-assurance)
14. [Privacy & Data Governance](#privacy--data-governance)
15. [Limitations & Future Roadmap](#limitations--future-roadmap)

---

## Project Overview

Early detection of learning differences is critical for timely educational intervention. Traditional diagnostic evaluations are often costly and delayed until late elementary school. 

The **Smart Learning Disability Screening System** bridges this gap by providing:
- **Child-Friendly Multi-Stage Assessments:** Engaging single-task screens that avoid cognitive fatigue.
- **Real-Time Word-by-Word Speech Alignment:** Analyzes oral reading using browser speech recognition and dynamic-programming sequence alignment.
- **OpenCV Computer Vision & Kinematics:** Analyzes actual canvas handwriting for stroke smoothness, baseline straightness, spatial spacing, and letter-form structure (such as b/d reversal).
- **Explainable Parent/Teacher Dashboards:** Delivers transparent insights explaining *what was observed*, *what it means*, and *suggested next steps*.

---

## Key Features

- **Automated Age-Adaptive Difficulty:** Age automatically configures task difficulty without manual selection.
- **Zero Raw Audio Storage:** Evaluates speech events and word alignment on-the-fly, protecting child voice privacy.
- **Microphone Integration with Visual Badges:** Real-time visual feedback categorizing words as correct, substituted, omitted, or repeated.
- **Computer Vision Structural Analysis:** OpenCV contours and Hu moments assess letter morphology alongside stroke acceleration/jerk.
- **Longitudinal History & Audit Logs:** SQLite database records session breakdowns and enables progress tracking over time.

---

## Age-Adaptive Architecture (Ages 5–12)

The system organizes content into four difficulty tiers, automatically determined from the child's recorded profile:

| Tier | Age Range | Dyslexia Screening Focus | Dysgraphia Handwriting Focus |
| :--- | :--- | :--- | :--- |
| **Beginner** | 5–6 years | Confusable letters (`b`/`d`, `p`/`q`), 3-letter CVC sight words, 5-word reading sentences, first-sound matching | Uppercase letters, lowercase `b`/`d`, 3-letter words (`cat`, `sun`), 2-word phrase (`red ball`) |
| **Elementary** | 7–8 years | High-frequency sight vocabulary, rhyming sounds, 10-word compound sentences, 4-letter spelling | Confusable lowercase letters (`d`, `p`), elementary words (`tree`, `happy`), short phrase (`my green garden`) |
| **Intermediate** | 9–10 years | Multi-syllabic words, multi-clause reading passages, phoneme deletion, 6-letter spelling | Lowercase letters (`q`, `m`), multi-syllable words (`school`, `friend`), guided sentence copying |
| **Advanced** | 11–12 years | Academic vocabulary, complex paragraph comprehension, morphology, advanced spelling | Challenging letters (`f`, `k`), abstract words (`mountain`, `journey`), full sentence with consistency evaluation |

---

## Dyslexia 7-Stage Screening Workflow

To provide comprehensive screening without overwhelming the child, the dyslexia assessment proceeds through 7 guided stages:

```
[Stage 1: Letter Identification]
       │
       ▼
[Stage 2: Single Word Recognition]
       │
       ▼
[Stage 3: Oral Reading with Microphone Alignment]
       │
       ▼
[Stage 4: Phonological Awareness (Rhyme, First/Last Sound)]
       │
       ▼
[Stage 5: Word Spelling Analysis]
       │
       ▼
[Stage 6: Reading Comprehension]
       │
       ▼
[Stage 7: Rapid Naming (RAN Speed)]
       │
       ▼
[Multi-Feature ML Classification & Explainable Report]
```

1. **Letter Recognition:** Identifies letter-sound correspondence and common mirror-image letter reversals (`b`/`d`, `p`/`q`, `m`/`n`, `u`/`v`).
2. **Word Recognition:** Evaluates single-word decoding accuracy and response time.
3. **Oral Reading:** Word-by-word microphone alignment comparing spoken transcript against target passage.
4. **Phonological Awareness:** Tests sound segmentation, blending, rhyming, and phoneme deletion.
5. **Spelling:** Evaluates letter order, transpositions, omissions, and extra letters.
6. **Reading Comprehension:** Tests recall and contextual understanding from short age-appropriate passages.
7. **Rapid Naming (RAN):** Measures lexical retrieval automaticity.

---

## Dysgraphia 5-Task Handwriting Workflow

The dysgraphia assessment presents a clean, uncluttered canvas environment:

- **Task 1:** Write an age-appropriate target letter.
- **Task 2:** Write a second target letter (e.g., assessing loop orientation).
- **Task 3:** Write a simple target word.
- **Task 4:** Write a second word.
- **Task 5:** Copy an age-appropriate phrase or sentence.

Child screen shows only:
- Instruction prompt (e.g. `Write this: cat`)
- Large interactive canvas with dual writing guide lines
- `Clear` button and `Next Task` button

---

## Speech Recognition & Alignment Engine

Located in `backend/services/speech_service.py`:
- Utilizes the browser **Web Speech API** (`SpeechRecognition` / `webkitSpeechRecognition`) for local, zero-cost processing.
- Implements a dynamic-programming sequence alignment algorithm based on SequenceMatcher / Levenshtein distance.
- Categorizes spoken tokens into:
  - **Correct:** Spoken word matches target.
  - **Substitution:** Word replaced (e.g., `little` -> `liddle` or `dog` -> `puppy`).
  - **Omission:** Target word was skipped.
  - **Insertion:** Unprompted extra word spoken.
  - **Repetition:** Same word repeated immediately (indicating re-reading or hesitation).
  - **Hesitation:** Pauses greater than 1.5 seconds.
- Reversal confusion detection identifies invert patterns such as `was`/`saw`, `on`/`no`, and `bad`/`dad`.

---

## Computer Vision & Stroke Kinematics

Located in `backend/vision/handwriting_analyzer.py`:
- **OpenCV Contour & Morphological Processing:**
  - Ink fill density and bounding box aspect ratio.
  - Number of connected components and contour complexity.
  - **Scale & Rotation Invariant Hu Moments:** Log-transformed moments capturing geometric character archetype.
  - **Letter-Form Mismatch Detection:** Checks horizontal mass asymmetry to flag reversed letter loops (e.g., loop on left vs right for `b` vs `d`).
- **Kinematic Feature Extraction:**
  - Normalized stroke speed and speed irregularity.
  - Stroke smoothness derived from speed and directional change angle variance.
  - Pen lifts and inter-stroke pause durations.
  - Character size consistency (coefficient of variation of cluster heights).
  - Inter-word spacing consistency.
  - Baseline straightness (standard deviation of vertical baseline alignment).

---

## Machine Learning & Scoring Pipeline

1. **Random Forest Classifier (Dyslexia):**
   - Trained on 1,000 multi-class records with 11 standardized features: `Age`, `Reading_Time_sec`, `Reading_Speed_WPM`, `Reading_Accuracy`, `Word_Error_Count`, `Letter_Reversal_Count`, `Spelling_Accuracy`, `Comprehension_Score`, `Avg_Response_Time_ms`, `Hesitation_Count`, and `Confidence_Score`.
   - Produces balanced 4-class probabilities: **Normal**, **Mild**, **Moderate**, and **Severe**.
2. **Calibrated Explainable Rubric (Dysgraphia):**
   - Synthesizes kinematic and CV measures into an explainable 0–100 screening score covering writing fluency, motor control, spatial organization, visual similarity, and letter-form accuracy.
   - Maps bounded score to risk categories with transparent mathematical contribution breakdowns.

---

## System Architecture & Database

```
Smart_LD_Detection_Age_5_10_Enhanced/
├── backend/
│   ├── main.py                     # FastAPI REST app & route controllers
│   ├── config.py                   # Environment settings & age-tier configuration
│   ├── database.py                 # SQLAlchemy ORM (User, AssessmentSession, Assessment)
│   ├── predictor.py                # Random Forest inference & rubric scoring
│   ├── features.py                 # Feature extraction & backward compatibility
│   ├── train_model.py              # Model training script with cross-validation
│   ├── build_test_banks.py         # Test bank generation script
│   ├── requirements.txt            # Python dependencies
│   ├── services/
│   │   ├── speech_service.py       # Speech DP alignment & error taxonomy
│   │   ├── dyslexia_service.py     # 7-stage dyslexia screening lifecycle
│   │   └── dysgraphia_service.py   # 5-task dysgraphia screening lifecycle
│   ├── vision/
│   │   └── handwriting_analyzer.py # OpenCV contours, Hu moments, kinematics
│   ├── models/                     # Saved joblib classifiers & scalers
│   └── data/                       # Test banks & training datasets
├── frontend/
│   ├── index.html                  # Single-page application entry point
│   ├── css/
│   │   └── style.css               # Modern educational platform design system
│   └── js/
│       └── app.js                  # Frontend controllers, canvas, & Web Speech API
├── tests/
│   └── test_backend.py             # Pytest automated test suite (100% passing)
├── .gitignore
└── README.md
```

---

## REST API Documentation

### Child Profile Management
- `POST /api/users`: Creates a new child profile (`name`, `age` [5–12]).
- `GET /api/users`: Lists all child profiles with their difficulty tier.
- `GET /api/users/{id}`: Returns profile details.
- `PUT /api/users/{id}`: Updates name or age.

### Dyslexia Screening
- `POST /api/dyslexia/session/start`: Initializes a randomized 7-stage session for the child's age tier.
- `POST /api/dyslexia/align-speech`: Real-time word-by-word speech alignment with error classification.
- `POST /api/dyslexia/session/submit`: Evaluates full session, runs Random Forest model, and generates explainable report.

### Dysgraphia Screening
- `POST /api/dysgraphia/session/start`: Initializes a 5-task handwriting session.
- `POST /api/dysgraphia/analyze-task`: Analyzes a single canvas sample with OpenCV & kinematics.
- `POST /api/dysgraphia/session/submit`: Evaluates all 5 tasks and generates multi-component rubric score.

### Results & History
- `GET /api/users/{id}/history`: Returns complete longitudinal assessment sessions.
- `GET /api/health`: System health and age support metadata.

Interactive Swagger documentation is available at `http://localhost:8000/docs`.

---

## Installation & Windows Setup Guide

### Prerequisites
- **Python 3.10 to 3.14** installed on Windows.
- **Google Chrome** or **Microsoft Edge** (recommended for Web Speech API microphone support).

### Option A: One-Click Launcher (Easiest)
Simply double-click **`run.bat`** in the project root, or execute in PowerShell:
```powershell
.\run.ps1
```
This automatically starts the FastAPI backend, starts the frontend HTTP server, and opens **`http://localhost:5500`** in your browser!

### Option B: Manual Terminal Execution

#### 1. Backend Setup
Open PowerShell or Command Prompt:

```powershell
# Navigate to backend directory
cd backend

# Install required dependencies (if not already installed)
pip install -r requirements.txt

# Start FastAPI backend server
python -m uvicorn main:app --reload --port 8000
```
The backend is now live at: `http://localhost:8000` (API Docs: `http://localhost:8000/docs`).

#### 2. Frontend Setup
In a second terminal window:

```powershell
# Navigate to frontend directory
cd frontend

# Serve frontend using Python's built-in HTTP server
python -m http.server 5500
```

Open your browser and navigate to:
**`http://localhost:5500`**

*(Note: Serving via `http://localhost:5500` rather than opening `file:///` is required for browser microphone permissions).*

### Option C: Unified Single-Port Execution (Recommended for Staging / Self-Host)
The FastAPI backend automatically serves the frontend on the same port:

```powershell
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```
Open **`http://localhost:8000`** — both frontend UI and REST API run together seamlessly.

### Option D: Docker Container Deployment (Production)

Build and run the production Docker container:

```bash
# 1. Build Docker image
docker build -t smart-ld-detection .

# 2. Run container
docker run -p 8000:8000 --name smart-ld smart-ld-detection
```

Navigate to **`http://localhost:8000`** in your browser. All models, backend endpoints, and frontend assets are packaged inside the container.

---

## Presentation & Demonstration Workflow

For project viva / presentations, follow this smooth workflow:

1. **Dashboard:**
   - Open `http://localhost:5500`.
   - Select or click **"+ New Child"** to create a profile (e.g. `Aarav`, Age `8`).
   - Point out that the difficulty badge automatically updates to **Elementary (Ages 7–8)** without manual difficulty selection.
2. **Dyslexia Screening Demonstration:**
   - Click **"Start Dyslexia Assessment"**.
   - **Stage 1 (Letters):** Point out confusable letters (`b`/`d`, `p`/`q`).
   - **Stage 2 (Words):** Read target sight words.
   - **Stage 3 (Microphone Oral Reading):**
     - Click the large 🎙 **Microphone** button. Allow mic permission.
     - Read the passage aloud. Point out the live word-by-word alignment chips (green = correct, yellow = substitution/pause, red = omission).
     - *(Note: If presenting in a noisy room, click "Demo Speech" to demonstrate simulated acoustic alignment).*
   - **Stages 4–7:** Complete phonological awareness, spelling, comprehension, and rapid naming.
   - Click **"Finish Screening"**.
3. **Results & Report:**
   - Observe the circular confidence gauge and 4-class probability breakdown.
   - Show the explainable report:
     - **"What We Observed"**: Plain-language errors and hesitations.
     - **"What This Means"**: Developmental interpretation.
     - **"Suggested Next Steps"**: Non-clinical guidance recommending consultation with educators.
4. **Dysgraphia Screening Demonstration:**
   - Return to Dashboard and launch **Dysgraphia Screening**.
   - Complete the 5 tasks on the large canvas using stylus, touch, or mouse:
     - Task 1: Letter `d`
     - Task 2: Letter `p`
     - Task 3: Word `tree`
     - Task 4: Word `happy`
     - Task 5: Sentence `my green garden`
   - Notice the live pointer pressure/device indicator and stroke guidelines.
   - Click **"Finish Assessment"** to inspect the Computer Vision and kinematic analysis.
5. **History & Longitudinal Audit:**
   - Click **"Assessment History"** in the sidebar to show both completed assessments recorded with timestamps and full report viewing.

---

## Testing & Quality Assurance

Run the automated test suite covering all APIs, speech alignment, vision processing, and age constraints:

```powershell
python -m pytest tests/test_backend.py -v
```

All 8 tests pass with 100% test coverage of core workflows:
- `test_health`: API metadata and screening disclaimer validation.
- `test_user_lifecycle_and_age_bounds`: Profile CRUD and strict 5–12 age enforcement.
- `test_speech_word_alignment_and_error_taxonomy`: Dynamic programming alignment and error taxonomy.
- `test_speech_reversal_detection`: Word-order and letter reversal detection.
- `test_dyslexia_session_flow`: Complete 7-stage session execution.
- `test_handwriting_cv_task_analysis`: OpenCV fill density, contours, Hu moments, and letter form checks.
- `test_dysgraphia_session_flow`: Complete 5-task dysgraphia execution.
- `test_user_history_persistence`: Audit trail retrieval and backwards compatibility.

---

## Privacy & Data Governance

1. **No Voice Recordings Stored:** Audio is transcribed in real-time within the browser. No raw audio files are uploaded or stored on disk.
2. **Minimal Data Collection:** Only child first name, age, and derived kinematic/lexical metrics are stored.
3. **Local Database:** All data resides within local SQLite (`backend/ld_detection.db`).
4. **No External Paid API Dependencies:** Entirely self-contained without mandatory cloud API keys.

---

## Limitations & Future Roadmap

- **Device Sensitivity:** Canvas handwriting capture using a mouse is an approximation of handwriting; a calibrated stylus tablet (Apple Pencil, Wacom, or Surface Pen) yields the highest kinematic accuracy.
- **Acoustic Environments:** Browser Web Speech API accuracy can vary with ambient background noise; testing in a quiet environment is recommended.
- **Non-Clinical Nature:** This application screens for observable indicators and should not be used as a standalone basis for educational placement or clinical diagnosis.
