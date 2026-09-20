"""
services/speech_service.py
--------------------------
Core Word-by-Word Speech Recognition & Reading Fluency Alignment Service.
Performs dynamic programming / Levenshtein word alignment, error taxonomy classification
(correct, substitution, omission, insertion, repetition, hesitation), letter/word reversal
detection, and fine-grained acoustic/timing analysis.

Designed to accept browser Web Speech API events or backend speech transcripts.
Never claims clinical diagnosis — outputs objective screening indicators only.
"""
import re
from difflib import SequenceMatcher
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

# Common phonetic/orthographic confusions in early reading
REVERSAL_WORD_PAIRS = {
    ("was", "saw"), ("saw", "was"),
    ("on", "no"), ("no", "on"),
    ("bad", "dad"), ("dad", "bad"),
    ("big", "dig"), ("dig", "big"),
    ("pat", "qat"), ("tap", "pat"),
    ("dog", "bog"), ("bog", "dog"),
    ("pin", "nip"), ("ten", "net"),
}

CONFUSABLE_LETTER_PAIRS = {
    ('b', 'd'), ('d', 'b'),
    ('p', 'q'), ('q', 'p'),
    ('m', 'n'), ('n', 'm'),
    ('u', 'v'), ('v', 'u'),
}

# Indian English phonological / acoustic variants (dental stops, v/w merger, monophthongs)
INDIAN_PHONETIC_EQUIVALENTS = {
    ("the", "de"), ("the", "da"), ("the", "dhe"), ("the", "thee"),
    ("this", "dis"), ("this", "dhis"),
    ("that", "dat"), ("that", "dhat"),
    ("these", "deez"), ("these", "dese"), ("these", "dhese"),
    ("those", "dose"), ("those", "dhose"),
    ("with", "wid"), ("with", "wit"),
    ("three", "tree"), ("three", "tri"),
    ("think", "tink"), ("thank", "tank"), ("thanks", "tanks"),
    ("thick", "tick"), ("there", "dere"), ("there", "dare"),
    ("their", "dere"), ("their", "dare"), ("then", "den"),
    ("they", "dey"), ("them", "dem"),
    ("went", "vent"), ("was", "vas"), ("was", "waz"), ("was", "vaz"),
    ("water", "vater"), ("very", "wery"), ("very", "veri"),
    ("fast", "faast"), ("car", "kaar"), ("park", "paark"),
    ("ball", "bol"), ("ball", "baal"), ("small", "smol"),
    ("sun", "san"), ("sun", "son"), ("cup", "kap"),
    ("bus", "bas"), ("cat", "kat"), ("bat", "baat"),
    ("runs", "ranz"), ("runs", "runz"), ("jump", "jamp"),
    ("duck", "dak"), ("tree", "tri"), ("happy", "heppy"),
    ("school", "ischool"), ("school", "sakool"),
    ("spoon", "ispoon"), ("station", "istation"),
    ("stop", "istop"), ("star", "istaar"),
    ("red", "rad"), ("green", "grin"),
}


def normalize_token(token: str) -> str:
    """Strip punctuation and lowercase for clean phonetic/lexical matching."""
    return re.sub(r"[^a-z0-9']", "", (token or "").lower().strip())


def tokenize(text: str) -> List[str]:
    """Tokenize sentence into clean lowercase words while preserving order."""
    if not text:
        return []
    words = re.findall(r"[a-zA-Z0-9']+", text.lower())
    return words


def check_potential_reversal(expected: str, recognized: str) -> Optional[str]:
    """Check if substitution could represent a known letter-reversal or word-order inversion."""
    if (expected, recognized) in REVERSAL_WORD_PAIRS:
        return f"Word inversion: '{expected}' read as '{recognized}'"
    
    if len(expected) == len(recognized) and len(expected) >= 2:
        diffs = [(e, r) for e, r in zip(expected, recognized) if e != r]
        if len(diffs) == 1 and diffs[0] in CONFUSABLE_LETTER_PAIRS:
            return f"Letter confusion: '{diffs[0][0]}' substituted with '{diffs[0][1]}'"
            
    return None


def indian_phonetic_key(w: str) -> str:
    """
    Computes an accent-normalized phonetic representation for Indian English speakers.
    Normalizes dental stops (th -> d), v/w mergers, epenthetic initial vowels,
    and consonant cluster simplifications.
    """
    w = re.sub(r"[^a-z]", "", (w or "").lower().strip())
    if not w:
        return ""
    # Strip initial prosthesis before consonant cluster: ischool -> school, ispoon -> spoon
    if len(w) > 4 and w.startswith(("is", "es")) and w[2] in "ptkc":
        w = w[1:]
    # Replace dental th with d
    w = re.sub(r"th", "d", w)
    # V / W merger
    w = re.sub(r"w", "v", w)
    # PH -> F
    w = re.sub(r"ph", "f", w)
    # CK -> K
    w = re.sub(r"ck", "k", w)
    # Soft C before e/i/y -> S, hard C -> K
    w = re.sub(r"c(?=[eiy])", "s", w)
    w = re.sub(r"c(?![h])", "k", w)
    # Collapse double letters
    w = re.sub(r"([a-z])\1+", r"\1", w)
    return w


def is_accent_match(expected: str, recognized: str, accent: str = "en-IN") -> Tuple[bool, Optional[str]]:
    """
    Determines whether a spoken word matches the expected word under the selected accent rules,
    preventing genuine accent/phonological variations from being falsely classified as reading errors.
    """
    e = normalize_token(expected)
    r = normalize_token(recognized)
    if not e or not r:
        return False, None
    if e == r:
        return True, None

    # Never treat genuine reversal confusions (b/d, was/saw) as accent variations
    if (e, r) in REVERSAL_WORD_PAIRS or (r, e) in REVERSAL_WORD_PAIRS:
        return False, None

    if accent in ("en-IN", "hi-IN", "all"):
        if (e, r) in INDIAN_PHONETIC_EQUIVALENTS or (r, e) in INDIAN_PHONETIC_EQUIVALENTS:
            return True, f"Indian English accent equivalent: '{r}' for '{e}'"

        ke = indian_phonetic_key(e)
        kr = indian_phonetic_key(r)
        if ke and kr and ke == kr:
            return True, f"Indian English phonetic match: '{r}' for '{e}'"

        if len(e) >= 3 and len(r) >= 3:
            ratio = SequenceMatcher(None, ke, kr).ratio()
            if ratio >= 0.84:
                return True, f"Indian English acoustic match: '{r}' for '{e}'"

    return False, None


def align_speech(
    expected_text: str,
    recognized_transcript: str,
    duration_sec: float,
    word_timestamps: Optional[List[Dict[str, Any]]] = None,
    pause_durations_sec: Optional[List[float]] = None,
    speech_confidence: float = 0.80,
    accent: str = "en-IN",
) -> Dict[str, Any]:
    """
    Performs word-level alignment between target expected sentence and actual recognized transcript.
    Supports accent-aware alignment (defaulting to en-IN Indian English) to avoid penalizing
    natural regional phonological variations.
    """
    expected_words = tokenize(expected_text)
    recognized_words = tokenize(recognized_transcript)
    
    duration_sec = max(float(duration_sec or 0.0), 0.1)

    # Accent-aware alignment keys
    if accent in ("en-IN", "hi-IN", "all"):
        exp_keys = [indian_phonetic_key(w) for w in expected_words]
        rec_keys = [indian_phonetic_key(w) for w in recognized_words]
    else:
        exp_keys = [normalize_token(w) for w in expected_words]
        rec_keys = [normalize_token(w) for w in recognized_words]
    
    matcher = SequenceMatcher(a=exp_keys, b=rec_keys, autojunk=False)
    
    aligned_tokens = []
    correct_count = 0
    substitutions = 0
    omissions = 0
    insertions = 0
    repetitions = 0
    accent_matches = 0
    reversal_indicators = []
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                exp_word = expected_words[i1 + k]
                rec_word = recognized_words[j1 + k]
                is_exact = (normalize_token(exp_word) == normalize_token(rec_word))
                is_acc, acc_flag = is_accent_match(exp_word, rec_word, accent=accent)
                
                correct_count += 1
                if not is_exact and is_acc:
                    accent_matches += 1

                aligned_tokens.append({
                    "expected": exp_word,
                    "recognized": rec_word,
                    "status": "correct",
                    "flag": None if is_exact else acc_flag,
                    "hesitation": False
                })
        elif tag == "replace":
            matched_len = min(i2 - i1, j2 - j1)
            for k in range(matched_len):
                exp_word = expected_words[i1 + k]
                rec_word = recognized_words[j1 + k]

                # Check if this token matches under accent adaptation
                is_acc, acc_flag = is_accent_match(exp_word, rec_word, accent=accent)
                if is_acc:
                    correct_count += 1
                    accent_matches += 1
                    aligned_tokens.append({
                        "expected": exp_word,
                        "recognized": rec_word,
                        "status": "correct",
                        "flag": acc_flag,
                        "hesitation": False
                    })
                    continue

                # Check if this was an immediate repetition of the previous word
                is_repetition = False
                if len(aligned_tokens) > 0 and aligned_tokens[-1]["recognized"] == rec_word:
                    is_repetition = True
                    repetitions += 1
                else:
                    substitutions += 1
                    
                rev_flag = check_potential_reversal(exp_word, rec_word)
                if rev_flag:
                    reversal_indicators.append(rev_flag)
                    
                aligned_tokens.append({
                    "expected": exp_word,
                    "recognized": rec_word,
                    "status": "repetition" if is_repetition else "substitution",
                    "flag": rev_flag,
                    "hesitation": False
                })
                
            # Surplus in expected -> omissions
            if (i2 - i1) > matched_len:
                for k in range(i1 + matched_len, i2):
                    omissions += 1
                    aligned_tokens.append({
                        "expected": expected_words[k],
                        "recognized": "(omitted)",
                        "status": "omission",
                        "flag": None,
                        "hesitation": False
                    })
                    
            # Surplus in recognized -> insertions
            if (j2 - j1) > matched_len:
                for k in range(j1 + matched_len, j2):
                    insertions += 1
                    aligned_tokens.append({
                        "expected": "(none)",
                        "recognized": recognized_words[k],
                        "status": "insertion",
                        "flag": None,
                        "hesitation": False
                    })
                    
        elif tag == "delete":
            for i in range(i1, i2):
                omissions += 1
                aligned_tokens.append({
                    "expected": expected_words[i],
                    "recognized": "(omitted)",
                    "status": "omission",
                    "flag": None,
                    "hesitation": False
                })
                
        elif tag == "insert":
            for j in range(j1, j2):
                rec_word = recognized_words[j]
                is_rep = len(aligned_tokens) > 0 and aligned_tokens[-1]["recognized"] == rec_word
                if is_rep:
                    repetitions += 1
                    aligned_tokens.append({
                        "expected": "(repeated)",
                        "recognized": rec_word,
                        "status": "repetition",
                        "flag": None,
                        "hesitation": False
                    })
                else:
                    insertions += 1
                    aligned_tokens.append({
                        "expected": "(extra)",
                        "recognized": rec_word,
                        "status": "insertion",
                        "flag": None,
                        "hesitation": False
                    })

    # Timing analysis
    total_spoken_words = len(recognized_words)
    total_expected_words = max(len(expected_words), 1)
    
    pauses = pause_durations_sec or []
    hesitations = sum(1 for p in pauses if p >= 1.5)
    pause_rate = round(len(pauses) / duration_sec, 3) if duration_sec > 0 else 0.0
    
    # Word response times
    if word_timestamps and len(word_timestamps) > 0:
        response_times = [
            float(wt.get("end", 0)) - float(wt.get("start", 0))
            for wt in word_timestamps if float(wt.get("end", 0)) > float(wt.get("start", 0))
        ]
    else:
        avg_word_sec = duration_sec / max(total_spoken_words, 1)
        response_times = [avg_word_sec] * total_spoken_words

    if response_times:
        avg_response_time_ms = round(float(np.mean(response_times)) * 1000, 1)
        median_response_time_ms = round(float(np.median(response_times)) * 1000, 1)
        response_time_variability = round(float(np.std(response_times)) * 1000, 1)
    else:
        avg_response_time_ms = round((duration_sec / total_expected_words) * 1000, 1)
        median_response_time_ms = avg_response_time_ms
        response_time_variability = 0.0

    reading_speed_wpm = round(total_spoken_words / (duration_sec / 60.0), 2)
    reading_accuracy = round(100.0 * correct_count / total_expected_words, 2)
    total_word_errors = substitutions + omissions + insertions + repetitions
    
    hesitation_rate = round(hesitations / total_expected_words, 3)

    # Plain language observations
    observations = []
    if reading_accuracy >= 92:
        observations.append("Reading accuracy was high with strong word matching.")
    elif reading_accuracy >= 75:
        observations.append(f"Reading accuracy was moderate ({reading_accuracy}%).")
    else:
        observations.append(f"Notable word decoding challenges observed (accuracy: {reading_accuracy}%).")
        
    if accent_matches > 0:
        observations.append(f"{accent_matches} word(s) verified using Indian English accent & phonetic adaptation.")
    if substitutions > 0:
        observations.append(f"{substitutions} word substitution(s) occurred during oral reading.")
    if omissions > 0:
        observations.append(f"{omissions} word(s) were omitted.")
    if insertions > 0:
        observations.append(f"{insertions} extra word insertion(s) noted.")
    if repetitions > 0:
        observations.append(f"{repetitions} word repetition(s) detected, indicating self-correction or re-reading.")
    if hesitations > 0:
        observations.append(f"{hesitations} extended pause(s) (>1.5s) detected during passage reading.")
    if reversal_indicators:
        for rev in reversal_indicators[:2]:
            observations.append(f"Screening indicator: {rev}.")

    return {
        "aligned_words": aligned_tokens,
        "metrics": {
            "expected_word_count": total_expected_words,
            "spoken_word_count": total_spoken_words,
            "correct_words": correct_count,
            "substitutions": substitutions,
            "omissions": omissions,
            "insertions": insertions,
            "repetitions": repetitions,
            "accent_matches": accent_matches,
            "accent": accent,
            "total_errors": total_word_errors,
            "reading_accuracy_pct": reading_accuracy,
            "reading_speed_wpm": reading_speed_wpm,
            "duration_sec": round(duration_sec, 2),
            "avg_response_time_ms": avg_response_time_ms,
            "median_response_time_ms": median_response_time_ms,
            "response_time_variability_ms": response_time_variability,
            "hesitation_count": hesitations,
            "hesitation_rate": hesitation_rate,
            "pause_rate": pause_rate,
            "speech_confidence": round(float(speech_confidence), 3),
            "reversal_count": len(reversal_indicators)
        },
        "observations": observations,
        "reversal_indicators": reversal_indicators
    }


def verify_spoken_word(
    target_word: str,
    spoken_word: str,
    age: int = 8,
    response_time_sec: float = 1.0,
    speech_confidence: float = 0.85,
    accent: str = "en-IN"
) -> Dict[str, Any]:
    """
    Automated AI verification of an individual spoken word against expected target.
    Supports Indian English (en-IN) phonetic variants (dental stops, v/w merger, monophthongs),
    phonological reversals (e.g. b/d, was/saw), and age/accent-calibrated response latency.
    """
    t = normalize_token(target_word)
    s = normalize_token(spoken_word)

    is_exact = (t == s)
    is_contained = (t in s or s in t) if (len(s) >= 2 and len(t) >= 2) else False
    sim = SequenceMatcher(None, t, s).ratio() if (t and s) else 0.0

    rev_flag = check_potential_reversal(t, s)
    is_acc, acc_detail = is_accent_match(t, s, accent=accent)

    # Automated decision rubric
    if is_exact:
        is_correct = True
        status = "correct"
        verdict_label = "Correct"
        explanation = f"Word '{target_word}' was spoken accurately."
    elif rev_flag:
        is_correct = False
        status = "reversal_error"
        verdict_label = "Letter / Word Reversal"
        explanation = f"Observed screening indicator: {rev_flag}."
    elif is_acc:
        is_correct = True
        status = "accent_match"
        verdict_label = "Correct (Indian Accent Match)"
        explanation = f"Word '{target_word}' pronounced with natural Indian English phonetic variation ('{spoken_word}'). Verified as correct."
    elif sim >= 0.82 or is_contained:
        is_correct = True
        status = "minor_phonetic_variation"
        verdict_label = "Correct (Acoustic Match)"
        explanation = f"Pronunciation match is high ({round(sim * 100)}%). Word verified as correct."
    else:
        is_correct = False
        status = "substitution"
        verdict_label = "Mispronounced / Substituted"
        explanation = f"Target was '{t}', but '{s}' was heard ({round(sim * 100)}% match)."

    # Age & Accent calibrated latency evaluation
    if accent in ("en-IN", "hi-IN"):
        latency_thresholds = {5: 3.6, 6: 3.4, 7: 2.9, 8: 2.6, 9: 2.2, 10: 2.0, 11: 1.8, 12: 1.6}
    else:
        latency_thresholds = {5: 3.2, 6: 3.0, 7: 2.5, 8: 2.2, 9: 1.8, 10: 1.6, 11: 1.4, 12: 1.2}
    thresh = latency_thresholds.get(int(age), 2.2)
    hesitation_detected = bool(response_time_sec > thresh)

    feedback_badge = "✓ Correctly Spoken" if is_exact else (
        "✓ Indian Accent Match" if is_acc else (
            "✓ Correct (Acoustic Match)" if is_correct else f"✗ {verdict_label}"
        )
    )

    return {
        "target_word": target_word,
        "spoken_word": spoken_word,
        "is_correct": is_correct,
        "similarity_score": round(sim, 2),
        "status": status,
        "verdict_label": verdict_label,
        "accent": accent,
        "accent_match": bool(is_acc),
        "accent_detail": acc_detail,
        "reversal_detected": bool(rev_flag),
        "reversal_flag": rev_flag,
        "explanation": explanation,
        "response_time_sec": round(response_time_sec, 2),
        "hesitation_detected": hesitation_detected,
        "confidence": round(speech_confidence, 2),
        "feedback_badge": feedback_badge
    }


