import re
from pathlib import Path

def test_html_js_ids():
    base = Path(__file__).resolve().parent.parent
    with open(base / "frontend" / "js" / "app.js", "r", encoding="utf-8") as f:
        js = f.read()
    with open(base / "frontend" / "index.html", "r", encoding="utf-8") as f:
        html = f.read()

    ids_in_js = set(re.findall(r'getElementById\(["\']([^"\']+)["\']\)', js))
    ids_in_html = set(re.findall(r'id=["\']([^"\']+)["\']', html))

    # Dynamically injected by stage renderers
    dynamic_ids = {
        'letterItemsList', 'btnNextDysStage', 'spellInput', 'btnSubmitSpell',
        'btnFinishDyslexia', 'micWrap', 'micStatusText', 'liveSpeechChips',
        'btnToggleMic', 'btnSpeechDemo', 'btnWordRight', 'btnWordWrong',
        'audioWave', 'liveModelStatus', 'liveHesitationCounter', 'liveSpeedWpm',
        'singleWordMicStatus', 'liveAccuracyPct', 'liveReadTimer', 'liveWordProgress',
        'btnMicWord', 'liveModelPill', 'activeWordTarget', 'liveHeardTranscript',
        'liveMicConfidence', 'liveAccuracySummaryBanner', 'liveAccuracyBigNum',
        'liveAccuracyDetail', 'liveCorrectWordsBadge', 'liveErrorsWordsBadge',
        'liveWordProgressText', 'liveWordComparisonGrid', 'singleWordCompCard',
        'swTargetText', 'swHeardText', 'swResultBadge', 'btnSpeechDemoStruggle',
        'btnDemoCorrectWord', 'btnDemoErrorWord', 'swModelExplanation',
        'inputSingleWordManual', 'btnVerifySingleManual', 'inputManualSpoken', 'btnVerifyManualSpoken',
        'btnStartRapidTest', 'btnStopRapidTest', 'rapidTimerDisplay', 'rapidErrCountDisplay',
        'btnRapidErrMinus', 'btnRapidErrPlus', 'rapidSpeedStatus', 'rapidSummaryCard',
        'rapidSummaryDetails', 'btnDemoRapidTypical', 'btnDemoRapidSlow'
    }

    missing = ids_in_js - ids_in_html - dynamic_ids
    assert len(missing) == 0, f"Missing IDs in HTML: {missing}"
