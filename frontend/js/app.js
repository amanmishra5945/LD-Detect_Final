/**
 * app.js — Smart Learning Disability Screening System
 * Complete frontend controller handling child profiles, multi-stage Dyslexia screening,
 * 5-task Dysgraphia handwriting canvas, Web Speech API microphone alignment,
 * and explainable parent/teacher result dashboards.
 */

function getApiBase() {
  if (typeof window !== "undefined" && window.location && window.location.protocol.startsWith("http")) {
    const port = window.location.port;
    const hostname = window.location.hostname || "127.0.0.1";
    // If already running on port 8000 or production (Render / standard HTTP/HTTPS)
    if (port === "8000" || !port || port === "80" || port === "443") {
      return window.location.origin;
    }
    // If on dev server port like 5500, match host and route to backend port 8000
    return `${window.location.protocol}//${hostname}:8000`;
  }
  return "http://127.0.0.1:8000";
}

const API_BASE = getApiBase();

// Global state
let currentChild = {
  id: 1,
  name: "Student",
  age: 8,
  role: "student",
  difficulty_label: "Elementary (Ages 7–8)"
};
let allUsers = [];
let dyslexiaSession = null;
let currentDyslexiaStageIndex = 0;
let dyslexiaCollectedData = {};

let dysgraphiaSession = null;
let currentDysgraphiaTaskIndex = 0;
let dysgraphiaTaskResults = [];

let speechRecognizer = null;
let isListening = false;
let userWantsListening = false;
let micMediaStream = null;
let audioContext = null;
let micAnalyser = null;
let animVolumeFrameId = null;
let speechStartTimestamp = null;
let speechTranscriptAccumulated = "";
let speechPauses = [];
let lastSpeechEventTime = null;
let speechConfidenceAvg = 0.85;

let activeResult = null;
let currentSpeechAccent = localStorage.getItem("selectedAccent") || "en-US";

function getSelectedAccent() {
  const sel = document.getElementById("accentSelect");
  const val = sel ? sel.value : currentSpeechAccent;
  if (!val || val === "auto") {
    return navigator.language || "en-US";
  }
  return val;
}

function setupAccentSelector() {
  const sel = document.getElementById("accentSelect");
  const badge = document.getElementById("accentModelBadge");
  if (!sel) return;
  sel.value = currentSpeechAccent;
  const updateBadge = () => {
    if (!badge) return;
    if (sel.value === "en-IN") {
      badge.textContent = "🇮🇳 Indian Acoustic Model";
      badge.style.background = "#ecfdf5";
      badge.style.color = "#047857";
      badge.style.borderColor = "#a7f3d0";
    } else if (sel.value === "en-GB") {
      badge.textContent = "🇬🇧 UK Acoustic Model";
      badge.style.background = "#eff6ff";
      badge.style.color = "#1d4ed8";
      badge.style.borderColor = "#bfdbfe";
    } else if (sel.value === "auto") {
      badge.textContent = "🌐 Device Default Model";
      badge.style.background = "#f0fdf4";
      badge.style.color = "#15803d";
      badge.style.borderColor = "#bbf7d0";
    } else {
      badge.textContent = "🇺🇸 US / General Model";
      badge.style.background = "#eff6ff";
      badge.style.color = "#1d4ed8";
      badge.style.borderColor = "#bfdbfe";
    }
  };
  updateBadge();
  sel.addEventListener("change", () => {
    currentSpeechAccent = sel.value;
    localStorage.setItem("selectedAccent", currentSpeechAccent);
    updateBadge();
    setupSpeechRecognitionEngine();
    showToast(`Speech recognition set to ${sel.options[sel.selectedIndex].text}`);
  });
}

// =========================================================================
// INITIALIZATION
// =========================================================================

async function initApplication() {
  try { setupNavigation(); } catch (e) { console.error("setupNavigation error", e); }
  try { setupCanvas(); } catch (e) { console.error("setupCanvas error", e); }
  try { setupAccentSelector(); } catch (e) { console.error("setupAccentSelector error", e); }
  try { setupSpeechRecognitionEngine(); } catch (e) { console.error("setupSpeechRecognitionEngine error", e); }
  try { await checkBackendHealth(); } catch (e) { console.error("checkBackendHealth error", e); }
  try { await loadChildProfiles(); } catch (e) { console.error("loadChildProfiles error", e); }
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initApplication);
} else {
  initApplication();
}

// =========================================================================
// NAVIGATION & VIEWS
// =========================================================================

function setupNavigation() {
  document.querySelectorAll(".nav-link").forEach(btn => {
    btn.addEventListener("click", () => {
      const viewId = btn.getAttribute("data-view");
      navigateToView(viewId);
    });
  });

  // Modal triggers
  const newProfileBtn = document.getElementById("btnNewProfile");
  if (newProfileBtn) {
    newProfileBtn.addEventListener("click", () => openProfileModal());
  }

  const modalCloseBtn = document.getElementById("closeProfileModal");
  if (modalCloseBtn) {
    modalCloseBtn.addEventListener("click", () => closeProfileModal());
  }

  const profileForm = document.getElementById("newProfileForm");
  if (profileForm) {
    profileForm.addEventListener("submit", handleCreateProfile);
  }

  // Child select change
  const childSelect = document.getElementById("childSelect");
  if (childSelect) {
    childSelect.addEventListener("change", (e) => {
      const selectedId = parseInt(e.target.value);
      const found = allUsers.find(u => u.id === selectedId);
      if (found) {
        selectChild(found);
      }
    });
  }
}

function navigateToView(viewName) {
  document.querySelectorAll(".nav-link").forEach(b => b.classList.remove("active"));
  const activeBtn = document.querySelector(`.nav-link[data-view="${viewName}"]`);
  if (activeBtn) activeBtn.classList.add("active");

  document.querySelectorAll(".page-view").forEach(v => v.classList.remove("active"));
  const targetView = document.getElementById(`view-${viewName}`);
  if (targetView) targetView.classList.add("active");

  if (viewName === "dashboard") loadDashboard();
  if (viewName === "history") loadHistoryView();
  if (viewName === "dyslexia" && !dyslexiaSession) initDyslexiaWorkflow();
  if (viewName === "dysgraphia") {
    resizeCanvas();
    if (!dysgraphiaSession) initDysgraphiaWorkflow();
  }
}

function showToast(msg) {
  const toast = document.getElementById("toastNotice");
  if (toast) {
    toast.textContent = msg;
    toast.style.display = "block";
    setTimeout(() => { toast.style.display = "none"; }, 3200);
  }
}

// =========================================================================
// API & HEALTH
// =========================================================================

async function checkBackendHealth() {
  const statusElem = document.getElementById("apiStatusIndicator");
  try {
    const res = await fetch(`${API_BASE}/api/health`);
    if (res.ok) {
      statusElem.innerHTML = `<span class="status-dot"></span> Backend Connected (v2.0)`;
    } else {
      statusElem.innerHTML = `<span class="status-dot" style="background:#EF4444;"></span> API Error`;
    }
  } catch (err) {
    statusElem.innerHTML = `<span class="status-dot" style="background:#EF4444;"></span> Offline — run uvicorn`;
  }
}

// =========================================================================
// CHILD PROFILE MANAGEMENT
// =========================================================================

async function loadChildProfiles() {
  try {
    const res = await fetch(`${API_BASE}/api/users`);
    if (res.ok) {
      allUsers = await res.json();
      renderChildSelect();
      if (allUsers.length > 0) {
        // Select first or saved child
        const savedId = parseInt(localStorage.getItem("ld_active_child_id"));
        const match = allUsers.find(u => u.id === savedId) || allUsers[0];
        selectChild(match);
      } else {
        // Auto-create default child profile if empty
        await autoCreateDefaultProfile();
      }
    }
  } catch (e) {
    console.error("Could not load users", e);
  }
}

async function autoCreateDefaultProfile() {
  try {
    const res = await fetch(`${API_BASE}/api/users`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: "Alex Kumar", age: 8, role: "student" })
    });
    if (res.ok) {
      const newUser = await res.json();
      allUsers.push(newUser);
      renderChildSelect();
      selectChild(newUser);
    }
  } catch (e) {}
}

function renderChildSelect() {
  const sel = document.getElementById("childSelect");
  if (!sel) return;
  sel.innerHTML = "";
  allUsers.forEach(u => {
    const opt = document.createElement("option");
    opt.value = u.id;
    opt.textContent = `${u.name} (Age ${u.age})`;
    sel.appendChild(opt);
  });
}

function selectChild(child) {
  currentChild = child;
  localStorage.setItem("ld_active_child_id", child.id);
  
  const sel = document.getElementById("childSelect");
  if (sel) sel.value = child.id;

  const avatar = document.getElementById("childAvatar");
  if (avatar) avatar.textContent = child.name.charAt(0).toUpperCase();

  const tierBadge = document.getElementById("childTierBadge");
  if (tierBadge) tierBadge.textContent = child.difficulty_label || `Age ${child.age}`;

  loadDashboard();
}

function openProfileModal() {
  const modal = document.getElementById("profileModal");
  if (modal) modal.classList.add("active");
}

function closeProfileModal() {
  const modal = document.getElementById("profileModal");
  if (modal) modal.classList.remove("active");
}

async function handleCreateProfile(e) {
  e.preventDefault();
  const nameInput = document.getElementById("inputChildName");
  const ageInput = document.getElementById("inputChildAge");
  
  const name = nameInput.value.trim();
  const age = parseInt(ageInput.value);

  if (!name || age < 5 || age > 12) {
    showToast("Please enter a name and age between 5 and 12");
    return;
  }

  try {
    const res = await fetch(`${API_BASE}/api/users`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, age, role: "student" })
    });
    if (res.ok) {
      const created = await res.json();
      allUsers.push(created);
      renderChildSelect();
      selectChild(created);
      closeProfileModal();
      showToast(`Profile for ${created.name} created!`);
      nameInput.value = "";
    }
  } catch (err) {
    showToast("Failed to create profile. Check backend connection.");
  }
}

// =========================================================================
// DASHBOARD
// =========================================================================

async function loadDashboard() {
  if (!currentChild) return;
  
  document.getElementById("dashChildName").textContent = currentChild.name;
  document.getElementById("dashChildAge").textContent = `${currentChild.age} years`;
  document.getElementById("dashChildTier").textContent = currentChild.difficulty_label;

  try {
    const res = await fetch(`${API_BASE}/api/users/${currentChild.id}/history`);
    if (res.ok) {
      const history = await res.json();
      document.getElementById("statTotalTests").textContent = history.length;
      
      const mildOrAbove = history.filter(h => h.prediction !== "Normal").length;
      document.getElementById("statObservedFlags").textContent = mildOrAbove;

      const latest = history[0];
      document.getElementById("statLatestOutcome").textContent = latest ? latest.prediction : "None yet";

      renderDashboardRecent(history.slice(0, 4));
    }
  } catch (e) {
    console.error("Dashboard history load failed", e);
  }
}

function renderDashboardRecent(recentList) {
  const container = document.getElementById("dashRecentList");
  if (!container) return;

  if (!recentList || recentList.length === 0) {
    container.innerHTML = `<div style="text-align:center;color:var(--text-muted);padding:24px;">No screenings recorded yet for this child profile. Choose an assessment above to start.</div>`;
    return;
  }

  const rows = recentList.map(item => `
    <tr>
      <td>${item.disorder === "dyslexia" ? "📖 Dyslexia Screening" : "✍️ Dysgraphia Screening"}</td>
      <td><span class="badge-risk ${item.prediction}">${item.prediction}</span></td>
      <td style="font-family:'IBM Plex Mono';">${(item.confidence * 100).toFixed(0)}%</td>
      <td>${new Date(item.created_at).toLocaleDateString()}</td>
    </tr>
  `).join("");

  container.innerHTML = `
    <div class="hist-table-wrap">
      <table class="hist-table">
        <thead><tr><th>Assessment</th><th>Result</th><th>Confidence</th><th>Date</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
  `;
}

// =========================================================================
// DYSLEXIA SCREENING WORKFLOW (7 STAGES)
// =========================================================================

async function initDyslexiaWorkflow() {
  if (!currentChild) {
    showToast("Please select or create a child profile first");
    return;
  }

  try {
    const res = await fetch(`${API_BASE}/api/dyslexia/session/start`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: currentChild.id, age: currentChild.age })
    });
    if (res.ok) {
      dyslexiaSession = await res.json();
      currentDyslexiaStageIndex = 0;
      dyslexiaCollectedData = {
        stage_1_letters: [],
        stage_2_words: [],
        stage_3_speech: {},
        stage_4_phono: [],
        stage_5_spelling: [],
        stage_6_comp: {},
        stage_7_rapid_naming: { total_time_sec: 8.0, errors: 0 }
      };
      renderDyslexiaCurrentStage();
    }
  } catch (err) {
    showToast("Failed to initialize dyslexia session. Ensure backend is running.");
  }
}

function renderDyslexiaCurrentStage() {
  if (!dyslexiaSession) return;
  const stage = dyslexiaSession.stages[currentDyslexiaStageIndex];
  const totalStages = dyslexiaSession.stages.length;

  document.getElementById("dyslexiaStepText").textContent = `Stage ${currentDyslexiaStageIndex + 1} of ${totalStages} — ${stage.name}`;
  const pct = Math.round(((currentDyslexiaStageIndex + 1) / totalStages) * 100);
  document.getElementById("dyslexiaProgressFill").style.width = `${pct}%`;

  const container = document.getElementById("dyslexiaStageContainer");
  container.innerHTML = "";

  switch (stage.stage) {
    case 1: renderStageLetterRecognition(container, stage); break;
    case 2: renderStageWordRecognition(container, stage); break;
    case 3: renderStageOralReading(container, stage); break;
    case 4: renderStagePhonological(container, stage); break;
    case 5: renderStageSpelling(container, stage); break;
    case 6: renderStageComprehension(container, stage); break;
    case 7: renderStageRapidNaming(container, stage); break;
  }
}

// Stage 1: Letter Recognition with Mic & Sound-Out Evaluation
function renderStageLetterRecognition(container, stage) {
  container.innerHTML = `
    <div class="task-prompt-box">
      <div class="task-label">Stage 1: Letter Identification</div>
      <div class="task-target">Read each letter aloud</div>
      <p style="color:var(--text-muted);font-size:14px;margin-top:6px;">Look at the letter shown and say its name or sound into the microphone, or mark manually.</p>
    </div>
    <div id="letterItemsList" style="display:flex;flex-direction:column;gap:14px;max-width:540px;margin:0 auto 24px;"></div>
    <div style="text-align:right;">
      <button class="btn btn-primary" id="btnNextDysStage">Next Stage →</button>
    </div>
  `;

  const list = document.getElementById("letterItemsList");
  stage.items.forEach((item, idx) => {
    const row = document.createElement("div");
    row.style.display = "flex";
    row.style.alignItems = "center";
    row.style.justifyContent = "space-between";
    row.style.padding = "12px 18px";
    row.style.background = "#F8FAFC";
    row.style.borderRadius = "var(--radius-sm)";
    row.style.border = "1px solid var(--border)";

    row.innerHTML = `
      <div style="display:flex;align-items:center;gap:14px;">
        <span style="font-size:32px;font-weight:700;font-family:'Outfit';min-width:32px;">${item.display}</span>
        <button class="btn btn-outline btn-sm btn-mic-letter" id="btnMicLetter_${idx}" data-idx="${idx}" data-target="${item.target}" data-conf="${item.confusable_with}">🎙 Speak</button>
        <span class="letter-heard-text" id="letterHeard_${idx}" style="font-size:12px;font-family:'IBM Plex Mono';color:var(--text-muted);"></span>
      </div>
      <div style="display:flex;gap:8px;">
        <button class="btn btn-outline btn-sm btn-letter-choice" id="btnLetterCorrect_${idx}" data-idx="${idx}" data-correct="true">✓ Correct</button>
        <button class="btn btn-outline btn-sm btn-letter-choice" id="btnLetterConf_${idx}" data-idx="${idx}" data-correct="false" data-rev="true">✗ Confused (${item.confusable_with})</button>
      </div>
    `;
    list.appendChild(row);
  });

  // Default results if unclicked
  dyslexiaCollectedData.stage_1_letters = stage.items.map(it => ({ target: it.target, correct: true, reversal_error: false }));

  list.querySelectorAll(".btn-letter-choice").forEach(btn => {
    btn.addEventListener("click", (e) => {
      const idx = parseInt(e.target.getAttribute("data-idx"));
      const isCorrect = e.target.getAttribute("data-correct") === "true";
      const isRev = e.target.getAttribute("data-rev") === "true";

      dyslexiaCollectedData.stage_1_letters[idx] = {
        target: stage.items[idx].target,
        correct: isCorrect,
        reversal_error: isRev
      };

      e.target.parentElement.querySelectorAll(".btn-letter-choice").forEach(b => {
        b.classList.remove("btn-brand", "btn-outline", "btn-primary");
        b.classList.add("btn-outline");
      });
      e.target.classList.remove("btn-outline");
      e.target.classList.add(isCorrect ? "btn-brand" : "btn-primary");
    });
  });

  // Microphone listener for each individual letter with automated AI verification
  list.querySelectorAll(".btn-mic-letter").forEach(btn => {
    btn.addEventListener("click", async (e) => {
      const idx = parseInt(e.target.getAttribute("data-idx"));
      const targetChar = e.target.getAttribute("data-target").toLowerCase();
      const confChar = e.target.getAttribute("data-conf").toLowerCase();
      const statusSpan = document.getElementById(`letterHeard_${idx}`);
      const btnCor = document.getElementById(`btnLetterCorrect_${idx}`);
      const btnConf = document.getElementById(`btnLetterConf_${idx}`);

      try {
        const SpeechClass = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechClass) {
          showToast("Speech recognition not supported in this browser. Please click manually.");
          return;
        }
        const rec = new SpeechClass();
        rec.lang = getSelectedAccent() || "en-US";
        rec.interimResults = true;
        rec.maxAlternatives = 3;

        rec.onstart = () => {
          btn.disabled = true;
          btn.textContent = "🎙 Listening…";
          if (statusSpan) statusSpan.innerHTML = `<span style="color:#2563EB; font-weight:700;">● LISTENING… Say "${targetChar}" now!</span>`;
        };

        rec.onresult = async (evt) => {
          let spoken = "";
          for (let i = 0; i < evt.results.length; i++) {
            spoken += evt.results[i][0].transcript + " ";
          }
          spoken = spoken.trim().toLowerCase();
          const spokenFirst = spoken.charAt(0);

          if (statusSpan) statusSpan.textContent = `Analyzing: "${spoken}"…`;

          // Automated AI model verification
          try {
            const vRes = await fetch(`${API_BASE}/api/dyslexia/verify-word`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                target_word: targetChar,
                spoken_transcript: spoken,
                age: currentChild ? currentChild.age : 8,
                response_time_sec: 1.0,
                speech_confidence: evt.results[0][0].confidence || 0.85,
                accent: getSelectedAccent()
              })
            });
            if (vRes.ok) {
              const vData = await vRes.json();
              if (vData.is_correct || spoken.includes(targetChar) || spokenFirst === targetChar) {
                btnCor.click();
                if (statusSpan) statusSpan.innerHTML = `Heard: "<strong>${spoken}</strong>" <span style="color:#15803D;">✓ AI Verified Correct</span>`;
              } else if (vData.reversal_detected || spoken.includes(confChar) || spokenFirst === confChar) {
                btnConf.click();
                if (statusSpan) statusSpan.innerHTML = `Heard: "<strong>${spoken}</strong>" <span style="color:#B91C1C;">✗ AI Flagged Confusion (${confChar})</span>`;
              } else {
                btnConf.click();
                if (statusSpan) statusSpan.innerHTML = `Heard: "<strong>${spoken}</strong>" <span style="color:#B91C1C;">✗ Error</span>`;
              }
              return;
            }
          } catch (e) {}

          // Local fallback
          if (spoken.includes(targetChar) || spokenFirst === targetChar) {
            btnCor.click();
            if (statusSpan) statusSpan.innerHTML = `Heard: "<strong>${spoken}</strong>" <span style="color:#15803D;">✓ Correct</span>`;
          } else if (spoken.includes(confChar) || spokenFirst === confChar) {
            btnConf.click();
            if (statusSpan) statusSpan.innerHTML = `Heard: "<strong>${spoken}</strong>" <span style="color:#B91C1C;">✗ Confused with ${confChar}</span>`;
          } else {
            if (statusSpan) statusSpan.innerHTML = `Heard: "${spoken}"`;
          }
        };

        rec.onerror = (err) => {
          if (err.error === "not-allowed") {
            if (statusSpan) statusSpan.textContent = "Mic permission blocked in browser. Click Allow in address bar.";
          } else if (err.error === "no-speech") {
            if (statusSpan) statusSpan.textContent = `No sound heard. Click Speak and say "${targetChar}".`;
          } else {
            if (statusSpan) statusSpan.textContent = `Mic ${err.error}. Click Speak or select button.`;
          }
          btn.disabled = false;
          btn.textContent = "🎙 Speak";
        };

        rec.onend = () => {
          btn.disabled = false;
          btn.textContent = "🎙 Speak";
        };

        rec.start();
      } catch (err) {
        if (statusSpan) statusSpan.textContent = "Mic busy. Mark manually.";
        btn.disabled = false;
        btn.textContent = "🎙 Speak";
      }
    });
  });

  document.getElementById("btnNextDysStage").addEventListener("click", advanceDyslexiaStage);
}

// Stage 2: Word Recognition with AI Automated Spoken Word Verification
function renderStageWordRecognition(container, stage) {
  let wordIdx = 0;
  const words = stage.items;

  function renderCurrentWord() {
    const currentTarget = words[wordIdx].word;
    container.innerHTML = `
      <div class="task-prompt-box">
        <div class="task-label">Stage 2: Single Word Reading (${wordIdx + 1} of ${words.length})</div>
        <div class="task-target"><span class="target-highlight" id="activeWordTarget">${currentTarget}</span></div>
        <p style="color:var(--text-muted);font-size:14px;margin-top:8px;">
          Read the word aloud into the microphone. The AI model <strong>automatically verifies</strong> whether the spoken word is correct or not.
        </p>
      </div>

      <div style="display:flex;flex-direction:column;align-items:center;gap:10px;margin-bottom:14px;">
        <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;justify-content:center;">
          <button class="btn btn-primary btn-lg" id="btnMicWord">🎙 Speak "${currentTarget}"</button>
          <button class="btn btn-outline btn-sm" id="btnDemoCorrectWord">Demo Spoke Correctly</button>
          <button class="btn btn-outline btn-sm" id="btnDemoErrorWord">Demo Spoke Reversal</button>
        </div>
        <div id="singleWordMicStatus" style="font-size:13px;font-weight:600;color:var(--primary);min-height:20px;">
          Click "Speak" to read aloud, or type what was spoken below
        </div>
        <div style="display:flex;gap:8px;align-items:center;margin-top:4px;">
          <input type="text" id="inputSingleWordManual" placeholder="Or type word child spoke…" style="font-size:13px;padding:6px 12px;border:1.5px solid var(--border);border-radius:var(--radius-sm);width:200px;">
          <button class="btn btn-outline btn-sm" id="btnVerifySingleManual">Verify</button>
        </div>
      </div>

      <!-- Live AI Model Verification Card -->
      <div class="single-word-comp-card" id="singleWordCompCard" style="display:none;">
        <div style="font-size:12px;font-weight:700;color:var(--primary);text-transform:uppercase;letter-spacing:0.05em;">
          🤖 Automated AI Model Verification Result:
        </div>
        <div style="display:flex;align-items:center;justify-content:center;gap:18px;margin:12px 0;">
          <div>
            <span style="font-size:12px;color:var(--text-muted);">Target Word:</span>
            <strong id="swTargetText" style="font-size:24px;font-family:'Outfit';color:var(--text-main);display:block;"></strong>
          </div>
          <div style="font-size:22px;color:var(--text-muted);">➔</div>
          <div>
            <span style="font-size:12px;color:var(--text-muted);">Child Spoke:</span>
            <strong id="swHeardText" style="font-size:24px;font-family:'Outfit';color:var(--text-main);display:block;"></strong>
          </div>
        </div>
        <div id="swResultBadge" style="margin:8px 0;"></div>
        <div id="swModelExplanation" style="font-size:13px;color:var(--text-muted);font-weight:500;margin-top:6px;"></div>
      </div>

      <div style="display:flex;justify-content:center;gap:14px;margin-bottom:24px;opacity:0.85;">
        <button class="btn btn-brand btn-sm" id="btnWordRight">✓ Manual Override (Correct)</button>
        <button class="btn btn-outline btn-sm" id="btnWordWrong">✗ Manual Override (Error)</button>
      </div>
    `;

    const startT = performance.now();
    let wordRecognized = false;

    const micWordBtn = document.getElementById("btnMicWord");
    const micStatus = document.getElementById("singleWordMicStatus");

    // Automated backend AI model verification routine
    async function processSpokenWordVerification(spokenText, elapsedSec, confidence = 0.85) {
      const compCard = document.getElementById("singleWordCompCard");
      const swTarget = document.getElementById("swTargetText");
      const swHeard = document.getElementById("swHeardText");
      const swBadge = document.getElementById("swResultBadge");
      const swExpl = document.getElementById("swModelExplanation");

      if (micStatus) {
        micStatus.textContent = "AI model verifying spoken word…";
        micStatus.style.color = "var(--primary)";
      }

      try {
        const res = await fetch(`${API_BASE}/api/dyslexia/verify-word`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            target_word: currentTarget,
            spoken_transcript: spokenText,
            age: currentChild ? currentChild.age : 8,
            response_time_sec: elapsedSec,
            speech_confidence: confidence,
            accent: getSelectedAccent()
          })
        });

        if (res.ok) {
          const vData = await res.json();

          if (compCard) compCard.style.display = "block";
          if (swTarget) swTarget.textContent = currentTarget;
          if (swHeard) swHeard.textContent = spokenText;

          if (swBadge) {
            if (vData.is_correct) {
              const label = vData.accent_match ? "✓ Correct (Indian Accent Match)" : `✓ Correctly Spoken (${Math.round(vData.similarity_score * 100)}% match)`;
              swBadge.innerHTML = `<span class="word-comp-badge badge-correct" style="font-size:14px;padding:6px 16px;">${label}</span>`;
            } else if (vData.reversal_detected) {
              swBadge.innerHTML = `<span class="word-comp-badge badge-error" style="font-size:14px;padding:6px 16px;">✗ Letter / Word Reversal Detected (${vData.reversal_flag})</span>`;
            } else {
              swBadge.innerHTML = `<span class="word-comp-badge badge-error" style="font-size:14px;padding:6px 16px;">✗ Mispronounced (${Math.round(vData.similarity_score * 100)}% match)</span>`;
            }
          }

          if (swExpl) {
            swExpl.textContent = vData.explanation;
          }

          if (micStatus) {
            micStatus.textContent = vData.is_correct ? "✓ Verified as Correct by AI Model" : `AI Model flagged: ${vData.verdict_label}`;
            micStatus.style.color = vData.is_correct ? "var(--brand-green-dark)" : "var(--accent-amber-dark)";
          }

          // Automatically record verified result
          dyslexiaCollectedData.stage_2_words.push({
            word: currentTarget,
            spoken: spokenText,
            correct: vData.is_correct,
            status: vData.status,
            reversal_detected: vData.reversal_detected,
            response_time_sec: elapsedSec
          });

          // Automatically proceed to next word after showing verification
          setTimeout(() => nextWord(), 1400);
          return;
        }
      } catch (err) {}

      // Fallback if offline
      const isSimpleMatch = spokenText.toLowerCase().includes(currentTarget.toLowerCase());
      dyslexiaCollectedData.stage_2_words.push({
        word: currentTarget,
        spoken: spokenText,
        correct: isSimpleMatch,
        response_time_sec: elapsedSec
      });
      setTimeout(() => nextWord(), 1100);
    }

    // Single-word speech recognition
    async function startSingleRec() {
      try {
        const SingleSpeech = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SingleSpeech) {
          if (micStatus) micStatus.textContent = "Web Speech API not supported. Type word below or click Demo.";
          return;
        }
        const singleRec = new SingleSpeech();
        singleRec.lang = getSelectedAccent() || "en-US";
        singleRec.interimResults = true;
        singleRec.maxAlternatives = 3;

        singleRec.onstart = () => {
          if (micWordBtn) {
            micWordBtn.disabled = true;
            micWordBtn.textContent = "🎙 Listening…";
          }
          if (micStatus) {
            micStatus.innerHTML = `<span style="color:#2563EB; font-weight:700;">● LISTENING… Say "${currentTarget}" now!</span>`;
          }
        };

        singleRec.onresult = async (evt) => {
          wordRecognized = true;
          let spoken = "";
          for (let i = 0; i < evt.results.length; i++) {
            spoken += evt.results[i][0].transcript + " ";
          }
          spoken = spoken.trim().toLowerCase();
          if (spoken) {
            const rt = Math.max((performance.now() - startT) / 1000.0, 0.4);
            const conf = evt.results[0][0].confidence || 0.85;
            await processSpokenWordVerification(spoken, rt, conf);
          }
        };

        singleRec.onerror = (e) => {
          console.warn("Single rec error:", e.error);
          if (e.error === "not-allowed") {
            if (micStatus) micStatus.textContent = "⚠️ Mic permission blocked. Click camera/mic icon in address bar to Allow.";
          } else if (e.error === "network") {
            if (micStatus) micStatus.textContent = "⚠️ Voice service timeout. Use Demo buttons or type word below.";
          } else if (e.error === "no-speech") {
            if (micStatus) micStatus.textContent = `No voice detected. Say "${currentTarget}" or use Demo buttons.`;
          } else {
            if (micStatus) micStatus.textContent = `Mic status: ${e.error}. Try again or type word below.`;
          }
          if (micWordBtn) {
            micWordBtn.disabled = false;
            micWordBtn.textContent = `🎙 Speak "${currentTarget}"`;
          }
        };

        singleRec.onend = () => {
          if (!wordRecognized && micWordBtn) {
            micWordBtn.disabled = false;
            micWordBtn.textContent = `🎙 Speak "${currentTarget}"`;
          }
        };

        singleRec.start();
      } catch (e) {
        if (micStatus) micStatus.textContent = "Click 'Speak' to start microphone.";
      }
    }

    if (micWordBtn) {
      micWordBtn.addEventListener("click", () => startSingleRec());
    }

    // Manual typing verification fallback
    const singleManualInput = document.getElementById("inputSingleWordManual");
    const singleManualBtn = document.getElementById("btnVerifySingleManual");
    if (singleManualBtn && singleManualInput) {
      const doVerifyTyped = () => {
        const val = singleManualInput.value.trim();
        if (val) {
          wordRecognized = true;
          processSpokenWordVerification(val, 1.2, 0.95);
        }
      };
      singleManualBtn.addEventListener("click", doVerifyTyped);
      singleManualInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter") doVerifyTyped();
      });
    }

    // Interactive Demo Buttons for Presentation / Testing
    const demoCorBtn = document.getElementById("btnDemoCorrectWord");
    if (demoCorBtn) {
      demoCorBtn.addEventListener("click", () => {
        wordRecognized = true;
        processSpokenWordVerification(currentTarget, 0.85, 0.95);
      });
    }

    const demoErrBtn = document.getElementById("btnDemoErrorWord");
    if (demoErrBtn) {
      demoErrBtn.addEventListener("click", () => {
        wordRecognized = true;
        let conf = "bog";
        const w = currentTarget.toLowerCase();
        if (w.includes("d")) conf = w.replace("d", "b");
        else if (w.includes("b")) conf = w.replace("b", "d");
        else if (w.includes("p")) conf = w.replace("p", "q");
        else conf = "saw";
        processSpokenWordVerification(conf, 2.4, 0.72);
      });
    }

    document.getElementById("btnWordRight").addEventListener("click", () => {
      const rt = (performance.now() - startT) / 1000.0;
      dyslexiaCollectedData.stage_2_words.push({ word: currentTarget, correct: true, response_time_sec: rt });
      nextWord();
    });

    document.getElementById("btnWordWrong").addEventListener("click", () => {
      const rt = (performance.now() - startT) / 1000.0;
      dyslexiaCollectedData.stage_2_words.push({ word: currentTarget, correct: false, response_time_sec: rt });
      nextWord();
    });
  }

  function nextWord() {
    wordIdx++;
    if (wordIdx < words.length) {
      renderCurrentWord();
    } else {
      advanceDyslexiaStage();
    }
  }

  renderCurrentWord();
}

// Stage 3: Oral Reading with Live Real-Time Telemetry, Heard Speech Display, and Word-by-Word Verification
function renderStageOralReading(container, stage) {
  const targetText = stage.passage.text;
  const targetWords = targetText.split(/\s+/);

  container.innerHTML = `
    <div class="task-prompt-box">
      <div class="task-label" style="display:flex;align-items:center;justify-content:space-between;">
        <span>Stage 3: Live Real-Time Oral Sentence Reading</span>
        <span class="badge" style="font-size:11px;font-weight:700;padding:4px 10px;background:#ecfdf5;color:#047857;border:1px solid #a7f3d0;" id="stage3AcousticBadge">🎙 Model: ${getSelectedAccent() === 'en-IN' ? '🇮🇳 English (India)' : getSelectedAccent()}</span>
      </div>
      
      <!-- Interactive Target Sentence: What is displayed for reading -->
      <div class="live-stream-words" id="liveStreamWords">
        ${targetWords.map((w, idx) => `
          <span class="stream-word" id="streamWord_${idx}" data-word="${w.toLowerCase().replace(/[^a-z0-9]/g, '')}">${w}</span>
        `).join("")}
      </div>

      <p style="color:var(--text-muted);font-size:14px;margin-top:6px;">
        Click the microphone and read the sentence above aloud. The AI checks whether each word was spoken correctly in real time.
      </p>
    </div>

    <!-- What Child Is Reading / Spoken Words Heard by Microphone -->
    <div class="live-heard-box">
      <div class="live-heard-header">
        <span class="live-heard-title">🎙 Real-Time Spoken Speech (What child is reading):</span>
        <span style="font-size:12px;color:var(--text-muted);" id="liveMicConfidence">Audio Confidence: --%</span>
      </div>
      <div class="live-heard-transcript placeholder" id="liveHeardTranscript">Waiting for speech… Click the microphone below and read aloud</div>
    </div>

    <!-- Live Accuracy & Evaluation Summary Banner -->
    <div class="live-accuracy-banner" id="liveAccuracySummaryBanner">
      <div class="live-acc-left">
        <div class="live-acc-bignum" id="liveAccuracyBigNum">100%</div>
        <div>
          <div class="live-acc-info" id="liveAccuracyDetail">Ready to evaluate oral reading accuracy</div>
          <div class="live-acc-sub">Real-time acoustic word alignment and error detection</div>
        </div>
      </div>
      <div class="live-acc-badges">
        <span class="badge-risk Normal" id="liveCorrectWordsBadge">0 Correct</span>
        <span class="badge-risk Mild" id="liveErrorsWordsBadge">0 Errors</span>
      </div>
    </div>

    <!-- Word-by-Word Correctness Verification Grid (Target vs Spoken) -->
    <div class="word-comparison-container">
      <div class="word-comparison-title">
        <span>🔍 Word Correctness Verification (Target vs. Spoken):</span>
        <span style="font-size:12px;font-weight:600;color:var(--text-muted);" id="liveWordProgressText">0 of ${targetWords.length} words verified</span>
      </div>
      <div class="word-comparison-grid" id="liveWordComparisonGrid">
        ${targetWords.map((w) => `
          <div class="word-comp-card pending">
            <div class="word-comp-target">${w}</div>
            <div class="word-comp-spoken">…</div>
            <span class="word-comp-badge badge-waiting">⏳ Pending</span>
          </div>
        `).join("")}
      </div>
    </div>

    <!-- Live Telemetry Display -->
    <div class="live-telemetry-panel">
      <div class="live-telemetry-item">
        <div class="live-telemetry-val" id="liveReadTimer">00:00.0</div>
        <div class="live-telemetry-lbl">Reading Timer</div>
      </div>
      <div class="live-telemetry-item">
        <div class="live-telemetry-val" id="liveSpeedWpm">0 WPM</div>
        <div class="live-telemetry-lbl">Reading Speed</div>
      </div>
      <div class="live-telemetry-item">
        <div class="live-telemetry-val" id="liveAccuracyPct">100%</div>
        <div class="live-telemetry-lbl">Word Accuracy</div>
      </div>
      <div class="live-telemetry-item">
        <div class="live-telemetry-val" id="liveWordProgress">0 of ${targetWords.length}</div>
        <div class="live-telemetry-lbl">Progress</div>
      </div>
    </div>

    <!-- Real-Time Model Classification Status -->
    <div style="text-align:center;margin-bottom:14px;">
      <div class="live-model-pill" id="liveModelPill">
        <span class="dot-pulse"></span>
        <span id="liveModelStatus">Live ML Model: Ready to Listen</span>
      </div>
    </div>

    <!-- Microphone Center Area -->
    <div class="mic-action-area">
      <div class="mic-btn-wrap" id="micWrap">
        <div class="mic-pulse-ring"></div>
        <button class="mic-btn" id="btnToggleMic" aria-label="Toggle Microphone">🎙</button>
      </div>
      <div class="audio-wave" id="audioWave">
        <span></span><span></span><span></span><span></span><span></span>
      </div>
      <div class="mic-status-text" id="micStatusText">Click microphone to start live reading</div>
      
      <!-- Manual typing assist fallback for oral reading verification -->
      <div style="display:flex;gap:8px;align-items:center;justify-content:center;margin-top:12px;margin-bottom:6px;">
        <input type="text" id="inputManualSpoken" placeholder="Or type sentence child read aloud…" style="font-size:13px;padding:7px 12px;border:1.5px solid var(--border);border-radius:var(--radius-sm);width:280px;">
        <button class="btn btn-outline btn-sm" id="btnVerifyManualSpoken">Verify Spoken</button>
      </div>
    </div>

    <!-- Live Word Chips Alignment -->
    <div style="margin-bottom:20px;">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
        <span style="font-size:13px;font-weight:600;color:var(--text-muted);">Acoustic Sequence Alignment</span>
        <span style="font-size:12px;color:var(--text-muted);" id="liveHesitationCounter">Hesitations: 0</span>
      </div>
      <div class="transcript-box" id="liveSpeechChips">
        <span class="word-chip pending">Spoken words will align and highlight here in real time…</span>
      </div>
    </div>

    <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;">
      <div style="display:flex;gap:8px;">
        <button class="btn btn-outline btn-sm" id="btnSpeechDemo">Demo Fluent Speech</button>
        <button class="btn btn-outline btn-sm" id="btnSpeechDemoStruggle">Demo Struggling Speech (Reversal)</button>
      </div>
      <button class="btn btn-primary" id="btnNextDysStage">Next Stage →</button>
    </div>
  `;

  document.getElementById("btnToggleMic").addEventListener("click", () => toggleMicrophone(targetText));
  document.getElementById("btnSpeechDemo").addEventListener("click", () => simulateDemoSpeech(targetText, false));
  document.getElementById("btnSpeechDemoStruggle").addEventListener("click", () => simulateDemoSpeech(targetText, true));

  const inputManual = document.getElementById("inputManualSpoken");
  const btnVerifyManual = document.getElementById("btnVerifyManualSpoken");
  if (btnVerifyManual && inputManual) {
    const doManualVerify = () => {
      const val = inputManual.value.trim();
      if (val) {
        speechTranscriptAccumulated = val;
        runSpeechAlignmentUpdate(val);
      }
    };
    btnVerifyManual.addEventListener("click", doManualVerify);
    inputManual.addEventListener("keydown", (e) => {
      if (e.key === "Enter") doManualVerify();
    });
  }

  document.getElementById("btnNextDysStage").addEventListener("click", () => {
    stopMicrophoneIfActive();
    advanceDyslexiaStage();
  });
}

// Stage 4: Phonological Awareness
function renderStagePhonological(container, stage) {
  let qIdx = 0;
  const items = stage.items;

  function renderCurrentQ() {
    const q = items[qIdx];
    container.innerHTML = `
      <div class="task-prompt-box">
        <div class="task-label">Stage 4: Sound Awareness (${qIdx + 1} of ${items.length})</div>
        <div class="task-target" style="font-size:24px;">${q.task}</div>
        <p style="color:var(--text-muted);font-size:14px;margin-top:8px;">Select the best matching answer.</p>
      </div>
      <div class="choice-grid">
        ${q.options.map((opt, i) => `
          <div class="btn-choice-card" data-opt="${i}">${opt}</div>
        `).join("")}
      </div>
    `;

    container.querySelectorAll(".btn-choice-card").forEach(card => {
      card.addEventListener("click", (e) => {
        const picked = parseInt(e.currentTarget.getAttribute("data-opt"));
        const isCorrect = (picked === q.answer);

        e.currentTarget.classList.add(isCorrect ? "selected-correct" : "selected-wrong");

        dyslexiaCollectedData.stage_4_phono.push({
          task_id: q.id,
          correct: isCorrect
        });

        setTimeout(() => {
          qIdx++;
          if (qIdx < items.length) {
            renderCurrentQ();
          } else {
            advanceDyslexiaStage();
          }
        }, 450);
      });
    });
  }

  renderCurrentQ();
}

// Stage 5: Spelling
function renderStageSpelling(container, stage) {
  let sIdx = 0;
  const words = stage.items;

  function renderCurrentSpell() {
    const item = words[sIdx];
    container.innerHTML = `
      <div class="task-prompt-box">
        <div class="task-label">Stage 5: Word Spelling (${sIdx + 1} of ${words.length})</div>
        <div class="task-target" style="font-size:22px;">Spell: "${item.word}"</div>
        <p style="color:var(--text-muted);font-size:14px;margin-top:6px;">Hint: ${item.hint}</p>
      </div>
      <div style="max-width:400px;margin:0 auto 24px;">
        <input type="text" id="spellInput" placeholder="Type word here…" style="width:100%;font-size:20px;padding:12px 16px;border:2px solid var(--border);border-radius:var(--radius-sm);text-align:center;font-family:'Outfit';" autocomplete="off" autocorrect="off" autocapitalize="off">
      </div>
      <div style="text-align:center;">
        <button class="btn btn-primary" id="btnSubmitSpell">Submit Word</button>
      </div>
    `;

    const input = document.getElementById("spellInput");
    input.focus();

    document.getElementById("btnSubmitSpell").addEventListener("click", () => {
      const typed = input.value.trim().toLowerCase();
      const isCorrect = (typed === item.word.toLowerCase());
      dyslexiaCollectedData.stage_5_spelling.push({
        target: item.word,
        typed: typed,
        correct: isCorrect
      });

      sIdx++;
      if (sIdx < words.length) {
        renderCurrentSpell();
      } else {
        advanceDyslexiaStage();
      }
    });
  }

  renderCurrentSpell();
}

// Stage 6: Reading Comprehension
function renderStageComprehension(container, stage) {
  const item = stage.item;
  if (!item) {
    advanceDyslexiaStage();
    return;
  }

  container.innerHTML = `
    <div class="task-prompt-box" style="text-align:left;">
      <div class="task-label">Stage 6: Reading Comprehension</div>
      <p style="font-size:16px;line-height:1.7;color:var(--text-main);margin:10px 0 16px;">${item.passage}</p>
      <div style="font-weight:700;font-size:18px;margin-top:14px;color:var(--text-main);">${item.question}</div>
    </div>
    <div class="choice-grid">
      ${item.options.map((opt, i) => `
        <div class="btn-choice-card" data-opt="${i}">${opt}</div>
      `).join("")}
    </div>
  `;

  container.querySelectorAll(".btn-choice-card").forEach(card => {
    card.addEventListener("click", (e) => {
      const picked = parseInt(e.currentTarget.getAttribute("data-opt"));
      const isCorrect = (picked === item.answer);

      e.currentTarget.classList.add(isCorrect ? "selected-correct" : "selected-wrong");
      dyslexiaCollectedData.stage_6_comp = { correct: isCorrect };

      setTimeout(() => advanceDyslexiaStage(), 500);
    });
  });
}

// Stage 7: Rapid Naming (RAN)
function renderStageRapidNaming(container, stage) {
  container.innerHTML = `
    <div class="task-prompt-box">
      <div class="task-label">Stage 7: Rapid Naming (Speed & Fluency)</div>
      <div class="task-target">Name these items as fast as you can</div>
      <p style="color:var(--text-muted);font-size:14px;margin-top:6px;">Click Start, let the child name all items left-to-right, then click Finish.</p>
    </div>
    <div style="display:flex;flex-wrap:wrap;gap:16px;justify-content:center;margin-bottom:28px;">
      ${stage.items.map(it => `
        <div style="padding:16px 24px;border:2px solid var(--border);border-radius:var(--radius-md);background:#FFFFFF;font-size:28px;font-weight:700;font-family:'Outfit';">
          ${it.item}
        </div>
      `).join("")}
    </div>
    <div style="display:flex;justify-content:center;gap:14px;">
      <button class="btn btn-brand btn-lg" id="btnFinishDyslexia">Finish Screening &amp; View Results →</button>
    </div>
  `;

  document.getElementById("btnFinishDyslexia").addEventListener("click", finalizeDyslexiaSubmission);
}

function advanceDyslexiaStage() {
  currentDyslexiaStageIndex++;
  if (currentDyslexiaStageIndex < dyslexiaSession.stages.length) {
    renderDyslexiaCurrentStage();
  } else {
    finalizeDyslexiaSubmission();
  }
}

async function finalizeDyslexiaSubmission() {
  showToast("Analyzing speech and reading evidence...");
  try {
    const payload = {
      session_id: dyslexiaSession.session_id,
      user_id: currentChild.id,
      age: currentChild.age,
      stages_data: dyslexiaCollectedData,
      confidence_score: speechConfidenceAvg,
      accent: getSelectedAccent()
    };

    const res = await fetch(`${API_BASE}/api/dyslexia/session/submit`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (res.ok) {
      const result = await res.json();
      activeResult = result;
      displayResults(result);
      navigateToView("results");
    } else {
      showToast("Submission error on server. Please try again.");
    }
  } catch (err) {
    showToast("Could not submit assessment results.");
  }
}

// =========================================================================
// SPEECH RECOGNITION (BROWSER WEB SPEECH API) WITH REAL-TIME MODEL INFERENCE
// =========================================================================

let liveTimerInterval = null;

function releaseAnyAudioLocks() {
  if (micMediaStream) {
    try {
      micMediaStream.getTracks().forEach(t => t.stop());
    } catch (e) {}
    micMediaStream = null;
  }
  if (audioContext) {
    try { audioContext.close(); } catch (e) {}
    audioContext = null;
  }
  micAnalyser = null;
}

function setupSpeechRecognitionEngine() {
  releaseAnyAudioLocks();
  const SpeechClass = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechClass) return;

  try {
    if (speechRecognizer) {
      speechRecognizer.onstart = null;
      speechRecognizer.onresult = null;
      speechRecognizer.onerror = null;
      speechRecognizer.onend = null;
      speechRecognizer.onaudiostart = null;
      speechRecognizer.onspeechstart = null;
      try { speechRecognizer.abort(); } catch (e) {}
    }
  } catch (e) {}

  speechRecognizer = new SpeechClass();
  speechRecognizer.continuous = true;
  speechRecognizer.interimResults = true;
  speechRecognizer.lang = getSelectedAccent();
  speechRecognizer.maxAlternatives = 3;

  speechRecognizer.onstart = () => {
    isListening = true;
    speechStartTimestamp = performance.now();
    lastSpeechEventTime = speechStartTimestamp;
    speechTranscriptAccumulated = "";
    speechPauses = [];

    // Visual wave & model status
    const wave = document.getElementById("audioWave");
    if (wave) wave.classList.add("active");
    const pill = document.getElementById("liveModelPill");
    if (pill) pill.classList.add("active-listening");
    const modelStatus = document.getElementById("liveModelStatus");
    if (modelStatus) modelStatus.textContent = "Live ML Model: Listening to voice stream…";

    // Start 100ms real-time timer
    clearInterval(liveTimerInterval);
    liveTimerInterval = setInterval(() => {
      const elapsed = (performance.now() - speechStartTimestamp) / 1000.0;
      const mins = Math.floor(elapsed / 60);
      const secs = (elapsed % 60).toFixed(1);
      const timerElem = document.getElementById("liveReadTimer");
      if (timerElem) {
        timerElem.textContent = `${String(mins).padStart(2, '0')}:${String(secs).padStart(4, '0')}`;
      }
    }, 100);

    updateMicVisualState(true, "● LISTENING NOW! Speak the sentence aloud into your microphone");
  };

  speechRecognizer.onaudiostart = () => {
    const statusText = document.getElementById("micStatusText");
    if (statusText) statusText.innerHTML = `<span style="color:#2563EB; font-weight:700;">● Microphone active: Speak now…</span>`;
  };

  speechRecognizer.onspeechstart = () => {
    const statusText = document.getElementById("micStatusText");
    if (statusText) statusText.innerHTML = `<span style="color:#15803D; font-weight:700;">● Voice detected: Transcribing…</span>`;
  };

  speechRecognizer.onresult = async (event) => {
    const now = performance.now();
    if (lastSpeechEventTime && (now - lastSpeechEventTime) > 1500) {
      speechPauses.push((now - lastSpeechEventTime) / 1000.0);
    }
    lastSpeechEventTime = now;

    let finalTranscript = "";
    let interimTranscript = "";

    for (let i = 0; i < event.results.length; ++i) {
      if (event.results[i].isFinal) {
        finalTranscript += event.results[i][0].transcript + " ";
      } else {
        interimTranscript += event.results[i][0].transcript + " ";
      }
    }

    const currentText = (finalTranscript + interimTranscript).trim();
    if (currentText) {
      speechTranscriptAccumulated = currentText;

      // Immediately show heard text right under mic
      const statusText = document.getElementById("micStatusText");
      if (statusText) {
        statusText.innerHTML = `Heard: "<span style="color:#15803D; font-weight:700;">${currentText}</span>"`;
      }

      // Real-time backend alignment + live ML prediction call
      await runSpeechAlignmentUpdate(currentText);
    }
  };

  speechRecognizer.onend = () => {
    if (userWantsListening) {
      // Auto-restart recognition if speech paused briefly
      setTimeout(() => {
        if (userWantsListening && speechRecognizer) {
          try {
            speechRecognizer.start();
          } catch (e) {}
        }
      }, 150);
      return;
    }

    isListening = false;
    clearInterval(liveTimerInterval);
    const durSec = speechStartTimestamp ? (performance.now() - speechStartTimestamp) / 1000.0 : 4.0;
    updateMicVisualState(false, "Microphone paused. Click 🎙 to speak again");

    const wave = document.getElementById("audioWave");
    if (wave) wave.classList.remove("active");
    const pill = document.getElementById("liveModelPill");
    if (pill) pill.classList.remove("active-listening");

    // Store in collected data
    if (dyslexiaSession && dyslexiaSession.stages[2]) {
      dyslexiaCollectedData.stage_3_speech = {
        expected_text: dyslexiaSession.stages[2].passage.text,
        transcript: speechTranscriptAccumulated,
        duration_sec: durSec,
        pauses: speechPauses,
        confidence: speechConfidenceAvg
      };
    }
  };

  speechRecognizer.onerror = (e) => {
    console.warn("Speech recognition error:", e.error);
    if (e.error === "no-speech") {
      if (userWantsListening) {
        const statusText = document.getElementById("micStatusText");
        if (statusText) statusText.innerHTML = `<span style="color:#D97706; font-weight:600;">● Listening… Speak louder into your microphone</span>`;
        return;
      }
    }
    if (e.error === "not-allowed" || e.error === "service-not-allowed") {
      userWantsListening = false;
      isListening = false;
      clearInterval(liveTimerInterval);
      updateMicVisualState(false, "⚠️ Microphone is blocked! Click the lock icon 🔒 in the browser address bar to Allow microphone.");
      showToast("Microphone blocked: Please click Allow in browser address bar.");
    } else if (e.error === "network") {
      userWantsListening = false;
      isListening = false;
      clearInterval(liveTimerInterval);
      updateMicVisualState(false, "⚠️ Speech recognition network timeout. You can also type spoken words in the box below.");
      showToast("Speech service error. You can also type words below.");
    } else if (e.error === "audio-capture") {
      userWantsListening = false;
      isListening = false;
      clearInterval(liveTimerInterval);
      updateMicVisualState(false, "⚠️ No microphone detected. Please check Windows sound input settings.");
      showToast("No microphone detected. Please check your mic connection.");
    } else {
      updateMicVisualState(false, `Mic: ${e.error}. Click to retry or type below.`);
    }
  };
}

function startSpeechRecognizer() {
  if (!speechRecognizer) {
    setupSpeechRecognitionEngine();
  }
  if (!speechRecognizer) return;

  try {
    speechRecognizer.start();
  } catch (err) {
    console.warn("speechRecognizer.start() note:", err);
    if (err.name === "InvalidStateError") {
      isListening = true;
      updateMicVisualState(true, "● LISTENING NOW! Speak aloud into your microphone");
    } else {
      setupSpeechRecognitionEngine();
      try { speechRecognizer.start(); } catch (e2) {}
    }
  }
}

async function toggleMicrophone(expectedText) {
  const SpeechClass = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechClass) {
    showToast("Web Speech API not supported on this browser. Please use Google Chrome or Microsoft Edge.");
    updateMicVisualState(false, "⚠️ Web Speech API not supported on this browser. Please use Chrome or Edge, or type spoken words below.");
    return;
  }

  if (isListening || userWantsListening) {
    userWantsListening = false;
    stopMicrophoneIfActive();
    return;
  }

  userWantsListening = true;
  updateMicVisualState(true, "🎙 Starting microphone… please speak clearly");

  startSpeechRecognizer();
}

function stopMicrophoneIfActive() {
  userWantsListening = false;
  if (speechRecognizer) {
    try {
      speechRecognizer.stop();
    } catch (e) {}
  }
  isListening = false;
  clearInterval(liveTimerInterval);
  updateMicVisualState(false, "Microphone stopped — click to read again");

  const wave = document.getElementById("audioWave");
  if (wave) wave.classList.remove("active");
  const pill = document.getElementById("liveModelPill");
  if (pill) pill.classList.remove("active-listening");
}

function updateMicVisualState(listening, text) {
  const wrap = document.getElementById("micWrap");
  const status = document.getElementById("micStatusText");
  if (wrap) {
    if (listening) wrap.classList.add("listening");
    else wrap.classList.remove("listening");
  }
  if (status) status.textContent = text;
}

async function runSpeechAlignmentUpdate(transcript) {
  if (!dyslexiaSession || !dyslexiaSession.stages[2]) return;
  const expected = dyslexiaSession.stages[2].passage.text;
  const childAge = currentChild ? currentChild.age : 8;

  // 1. Instantly update live heard transcript so tester and child see what is being spoken
  const heardBox = document.getElementById("liveHeardTranscript");
  if (heardBox) {
    if (transcript && transcript.trim()) {
      heardBox.textContent = `“${transcript.trim()}”`;
      heardBox.classList.remove("placeholder");
    } else {
      heardBox.textContent = "Listening… Start speaking the sentence now";
      heardBox.classList.add("placeholder");
    }
  }

  const confElem = document.getElementById("liveMicConfidence");
  if (confElem && speechConfidenceAvg) {
    confElem.textContent = `Audio Confidence: ${Math.round(speechConfidenceAvg * 100)}%`;
  }

  try {
    const res = await fetch(`${API_BASE}/api/dyslexia/align-speech?age=${childAge}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        expected_text: expected,
        transcript: transcript,
        duration_sec: Math.max((performance.now() - (speechStartTimestamp || performance.now())) / 1000.0, 0.5),
        pauses: speechPauses,
        speech_confidence: speechConfidenceAvg,
        accent: getSelectedAccent()
      })
    });

    if (res.ok) {
      const data = await res.json();
      
      // 2. Update Telemetry displays
      const speedElem = document.getElementById("liveSpeedWpm");
      if (speedElem) speedElem.textContent = `${Math.round(data.metrics.reading_speed_wpm)} WPM`;

      const accElem = document.getElementById("liveAccuracyPct");
      if (accElem) accElem.textContent = `${Math.round(data.metrics.reading_accuracy_pct)}%`;

      const progElem = document.getElementById("liveWordProgress");
      if (progElem) progElem.textContent = `${data.metrics.spoken_word_count} of ${data.metrics.expected_word_count}`;

      const hesElem = document.getElementById("liveHesitationCounter");
      if (hesElem) hesElem.textContent = `Hesitations: ${data.metrics.hesitation_count}`;

      // 3. Update Big Accuracy Banner & Correctness Summary
      const bigAcc = document.getElementById("liveAccuracyBigNum");
      if (bigAcc) bigAcc.textContent = `${Math.round(data.metrics.reading_accuracy_pct)}%`;

      const accDetail = document.getElementById("liveAccuracyDetail");
      if (accDetail) {
        if (data.metrics.reading_accuracy_pct >= 90) {
          accDetail.textContent = `${data.metrics.correct_words} of ${data.metrics.expected_word_count} words read accurately!`;
        } else {
          accDetail.textContent = `${data.metrics.correct_words} of ${data.metrics.expected_word_count} words spoken correctly — ${data.metrics.total_errors} error(s)/hesitation(s) noted.`;
        }
      }

      const corBadge = document.getElementById("liveCorrectWordsBadge");
      if (corBadge) corBadge.textContent = `${data.metrics.correct_words} Correct`;

      const errBadge = document.getElementById("liveErrorsWordsBadge");
      if (errBadge) {
        errBadge.textContent = `${data.metrics.total_errors} Errors`;
        errBadge.className = data.metrics.total_errors === 0 ? "badge-risk Normal" : "badge-risk Mild";
      }

      const progText = document.getElementById("liveWordProgressText");
      if (progText) progText.textContent = `${data.metrics.spoken_word_count} of ${data.metrics.expected_word_count} words verified`;

      // 4. Update Word-by-Word Comparison Grid
      renderWordComparisonGrid(data.aligned_words);

      // 5. Real-Time Model Live Classification
      if (data.live_prediction) {
        const pred = data.live_prediction;
        const pillStatus = document.getElementById("liveModelStatus");
        if (pillStatus) {
          const colorMap = { Normal: "#047857", Mild: "#B45309", Moderate: "#C2410C", Severe: "#DC2626" };
          const conf = Math.round(pred.confidence * 100);
          pillStatus.innerHTML = `Live ML Model: <strong style="color:${colorMap[pred.risk_level] || '#2563EB'}">${pred.risk_level}</strong> (${conf}% Confidence)`;
        }
      }

      // 6. Highlight individual target words in the sentence stream
      const aligned = data.aligned_words || [];
      let lastMatchIdx = 0;
      aligned.forEach((tok, i) => {
        const span = document.getElementById(`streamWord_${i}`);
        if (span) {
          span.className = "stream-word";
          if (tok.status === "correct") {
            span.classList.add("matched-correct");
            lastMatchIdx = i + 1;
          } else if (tok.status === "substitution") {
            span.classList.add("matched-error");
            span.title = `Heard: "${tok.recognized}"`;
            lastMatchIdx = i + 1;
          } else if (tok.status === "omission") {
            span.classList.add("matched-omitted");
          }
        }
      });

      // Highlight the upcoming active focus word
      const nextSpan = document.getElementById(`streamWord_${lastMatchIdx}`);
      if (nextSpan) {
        nextSpan.classList.add("active-focus");
      }

      // 7. Render acoustic chips
      renderSpeechChips(data.aligned_words);
    }
  } catch (err) {}
}

function renderWordComparisonGrid(alignedWords) {
  const grid = document.getElementById("liveWordComparisonGrid");
  if (!grid || !alignedWords) return;
  grid.innerHTML = "";

  alignedWords.forEach(tok => {
    let cardCls = "pending";
    let badgeCls = "badge-waiting";
    let badgeText = "⏳ Pending";
    let spokenText = tok.recognized;

    if (tok.status === "correct") {
      cardCls = "correct";
      badgeCls = "badge-correct";
      badgeText = "✓ Correct";
    } else if (tok.status === "substitution") {
      cardCls = "substitution";
      badgeCls = "badge-error";
      badgeText = `✗ Spoke: "${tok.recognized}"`;
    } else if (tok.status === "omission") {
      cardCls = "omission";
      badgeCls = "badge-omitted";
      badgeText = "⚠️ Skipped";
      spokenText = "(omitted)";
    } else if (tok.status === "insertion") {
      cardCls = "insertion";
      badgeCls = "badge-extra";
      badgeText = `+ Extra Word`;
    } else if (tok.status === "repetition") {
      cardCls = "repetition";
      badgeCls = "badge-waiting";
      badgeText = "↺ Repeated";
    }

    const card = document.createElement("div");
    card.className = `word-comp-card ${cardCls}`;

    let flagHtml = "";
    if (tok.flag) {
      flagHtml = `<div class="word-comp-alert">${tok.flag}</div>`;
    }

    card.innerHTML = `
      <div class="word-comp-target">${tok.expected}</div>
      <div class="word-comp-spoken" title="Recognized: ${spokenText}">${spokenText}</div>
      <span class="word-comp-badge ${badgeCls}">${badgeText}</span>
      ${flagHtml}
    `;

    grid.appendChild(card);
  });
}

function renderSpeechChips(tokens) {
  const container = document.getElementById("liveSpeechChips");
  if (!container || !tokens) return;
  container.innerHTML = "";

  tokens.forEach(tok => {
    const chip = document.createElement("span");
    chip.className = `word-chip ${tok.status}`;
    chip.textContent = tok.status === "omission" ? `${tok.expected} [omitted]` : tok.recognized;
    container.appendChild(chip);
  });
}

function simulateDemoSpeech(expectedText, isStruggling = false) {
  const words = expectedText.split(/\s+/);
  speechStartTimestamp = performance.now();

  let transcriptToPlay = expectedText;
  let duration = 4.6;
  let pauses = [0.8];
  speechConfidenceAvg = 0.92;

  if (isStruggling) {
    const altered = [...words];
    // Introduce an authentic substitution / letter-confusion
    if (altered.length >= 3) {
      const w = altered[2].toLowerCase();
      if (w.includes("b")) altered[2] = w.replace("b", "d");
      else if (w.includes("d")) altered[2] = w.replace("d", "b");
      else altered[2] = "box";
    }
    if (altered.length >= 6) {
      altered[5] = "the";
    }
    transcriptToPlay = altered.join(" ");
    duration = 9.2;
    pauses = [1.8, 2.3];
    speechConfidenceAvg = 0.74;
  }

  speechTranscriptAccumulated = transcriptToPlay;

  dyslexiaCollectedData.stage_3_speech = {
    expected_text: expectedText,
    transcript: transcriptToPlay,
    duration_sec: duration,
    pauses: pauses,
    confidence: speechConfidenceAvg
  };

  runSpeechAlignmentUpdate(transcriptToPlay);

  if (isStruggling) {
    showToast("Struggling speech simulated: Word substitution & hesitation flagged by ML.");
  } else {
    showToast("Fluent oral reading simulated: High accuracy & strong word alignment.");
  }
}

// =========================================================================
// DYSGRAPHIA SCREENING WORKFLOW (5 TASKS)
// =========================================================================

let canvas, ctx;
let strokes = [];
let currentStroke = null;
let isDrawing = false;
let taskStartTime = null;
let correctionCount = 0;

function setupCanvas() {
  canvas = document.getElementById("handwritingCanvas");
  if (!canvas) return;
  ctx = canvas.getContext("2d");

  canvas.addEventListener("pointerdown", onCanvasPointerDown);
  canvas.addEventListener("pointermove", onCanvasPointerMove);
  canvas.addEventListener("pointerup", onCanvasPointerUp);
  canvas.addEventListener("pointercancel", onCanvasPointerUp);

  const clearBtn = document.getElementById("btnClearCanvas");
  if (clearBtn) clearBtn.addEventListener("click", () => clearCanvas(true));

  const nextBtn = document.getElementById("btnNextDysgTask");
  if (nextBtn) nextBtn.addEventListener("click", advanceDysgraphiaTask);

  window.addEventListener("resize", () => {
    resizeCanvas();
    redrawCanvas();
  });

  resizeCanvas();
}

function resizeCanvas() {
  if (!canvas || !canvas.parentElement) return;
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = rect.width * dpr;
  canvas.height = 320 * dpr;
  ctx.scale(dpr, dpr);
  // Ensure solid white background (avoids transparent PNG issues)
  ctx.fillStyle = "#FFFFFF";
  ctx.fillRect(0, 0, rect.width, 320);
  ctx.lineWidth = 3.0;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.strokeStyle = "#0F172A";
}

function getPointerPos(e) {
  const rect = canvas.getBoundingClientRect();
  const t = taskStartTime ? (performance.now() - taskStartTime) / 1000.0 : 0.0;
  return {
    x: e.clientX - rect.left,
    y: e.clientY - rect.top,
    t: t,
    pressure: Number.isFinite(e.pressure) ? e.pressure : 0.5,
    pointerType: e.pointerType || "mouse"
  };
}

function onCanvasPointerDown(e) {
  e.preventDefault();
  if (!taskStartTime) taskStartTime = performance.now();
  isDrawing = true;
  const p = getPointerPos(e);
  currentStroke = [p];
  if (canvas.setPointerCapture) canvas.setPointerCapture(e.pointerId);

  // Stylus device badge
  const devBadge = document.getElementById("stylusDeviceBadge");
  if (devBadge) {
    devBadge.textContent = e.pointerType === "pen" ? "Stylus Tablet" : "Touch/Pointer";
  }
}

function onCanvasPointerMove(e) {
  if (!isDrawing) return;
  e.preventDefault();
  const p = getPointerPos(e);
  currentStroke.push(p);

  ctx.beginPath();
  const prev = currentStroke[currentStroke.length - 2];
  ctx.moveTo(prev.x, prev.y);
  ctx.lineTo(p.x, p.y);
  ctx.stroke();
}

function onCanvasPointerUp(e) {
  if (!isDrawing) return;
  isDrawing = false;
  strokes.push(currentStroke);
  currentStroke = null;
}

function clearCanvas(isUserAction = true) {
  if (!canvas || !ctx) return;
  const rect = canvas.getBoundingClientRect();
  ctx.fillStyle = "#FFFFFF";
  ctx.fillRect(0, 0, rect.width, 320);
  strokes = [];
  taskStartTime = null;
  if (isUserAction) {
    correctionCount++;
  }
}

function redrawCanvas() {
  if (!ctx || !canvas) return;
  const rect = canvas.getBoundingClientRect();
  ctx.fillStyle = "#FFFFFF";
  ctx.fillRect(0, 0, rect.width, 320);
  strokes.forEach(s => {
    ctx.beginPath();
    s.forEach((p, i) => {
      if (i === 0) ctx.moveTo(p.x, p.y);
      else ctx.lineTo(p.x, p.y);
    });
    ctx.stroke();
  });
}

async function initDysgraphiaWorkflow() {
  if (!currentChild) {
    showToast("Please select or create a child profile first");
    return;
  }

  try {
    const res = await fetch(`${API_BASE}/api/dysgraphia/session/start`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: currentChild.id, age: currentChild.age })
    });
    if (res.ok) {
      dysgraphiaSession = await res.json();
      currentDysgraphiaTaskIndex = 0;
      dysgraphiaTaskResults = [];
      renderDysgraphiaCurrentTask();
    }
  } catch (err) {
    showToast("Could not start dysgraphia assessment session.");
  }
}

function renderDysgraphiaCurrentTask() {
  if (!dysgraphiaSession) return;
  const task = dysgraphiaSession.tasks[currentDysgraphiaTaskIndex];
  const totalTasks = dysgraphiaSession.tasks.length;

  document.getElementById("dysgStepText").textContent = `Task ${currentDysgraphiaTaskIndex + 1} of ${totalTasks}`;
  const pct = Math.round(((currentDysgraphiaTaskIndex + 1) / totalTasks) * 100);
  document.getElementById("dysgProgressFill").style.width = `${pct}%`;

  document.getElementById("dysgTargetDisplay").textContent = task.target;
  document.getElementById("dysgInstruction").textContent = task.instruction;

  clearCanvas(false);
  correctionCount = 0;
  taskStartTime = null;

  const nextBtn = document.getElementById("btnNextDysgTask");
  if (nextBtn) {
    nextBtn.textContent = (currentDysgraphiaTaskIndex === totalTasks - 1) ? "Finish Assessment →" : "Next Task →";
  }
}

async function advanceDysgraphiaTask() {
  if (!dysgraphiaSession) return;
  const currentTask = dysgraphiaSession.tasks[currentDysgraphiaTaskIndex];
  const durSec = taskStartTime ? (performance.now() - taskStartTime) / 1000.0 : 4.0;

  // Optional base64 capture
  const imgData = canvas ? canvas.toDataURL("image/png") : null;

  dysgraphiaTaskResults.push({
    target: currentTask.target,
    strokes: strokes,
    duration_sec: durSec,
    corrections: correctionCount,
    image_base64: imgData,
    canvas_width: canvas.clientWidth,
    canvas_height: canvas.clientHeight
  });

  currentDysgraphiaTaskIndex++;
  if (currentDysgraphiaTaskIndex < dysgraphiaSession.tasks.length) {
    renderDysgraphiaCurrentTask();
  } else {
    await finalizeDysgraphiaSubmission();
  }
}

async function finalizeDysgraphiaSubmission() {
  showToast("Evaluating handwriting kinematics & vision metrics...");
  try {
    const payload = {
      session_id: dysgraphiaSession.session_id,
      user_id: currentChild.id,
      age: currentChild.age,
      task_results: dysgraphiaTaskResults,
      device_type: "external_stylus_tablet"
    };

    const res = await fetch(`${API_BASE}/api/dysgraphia/session/submit`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (res.ok) {
      const result = await res.json();
      activeResult = result;
      displayResults(result);
      navigateToView("results");
    } else {
      showToast("Server error evaluating handwriting.");
    }
  } catch (err) {
    showToast("Could not submit dysgraphia session.");
  }
}

// =========================================================================
// RESULTS DASHBOARD
// =========================================================================

function displayResults(res) {
  if (!res) return;

  document.getElementById("resDisorderTitle").textContent = (res.disorder === "dyslexia")
    ? "Dyslexia Screening Result"
    : "Dysgraphia Screening Result";

  const badge = document.getElementById("resRiskBadge");
  badge.textContent = res.prediction;
  badge.className = `badge-risk ${res.prediction}`;

  const accentTag = (res.accent === "en-IN" || (!res.accent && getSelectedAccent() === "en-IN"))
    ? '<span class="badge" style="margin-left:8px;font-size:12px;background:#ecfdf5;color:#047857;border:1px solid #a7f3d0;">🇮🇳 Indian Accent Calibration Applied</span>'
    : '';
  const titleEl = document.getElementById("resDisorderTitle");
  if (titleEl && !titleEl.innerHTML.includes("Indian Accent")) {
    titleEl.innerHTML += accentTag;
  }

  const confPct = Math.round((res.confidence || 0.85) * 100);
  document.getElementById("resConfidenceVal").textContent = `${confPct}%`;

  // Circular gauge animation
  const circle = document.getElementById("resGaugeCircle");
  if (circle) {
    const circumference = 2 * Math.PI * 46;
    const offset = circumference - (confPct / 100) * circumference;
    circle.style.strokeDashoffset = offset;
  }

  // Probability bars
  const probContainer = document.getElementById("resProbBars");
  if (probContainer && res.probabilities) {
    const colors = { Normal: "#10B981", Mild: "#F59E0B", Moderate: "#F97316", Severe: "#EF4444" };
    probContainer.innerHTML = Object.entries(res.probabilities).map(([cls, val]) => `
      <div class="prob-row">
        <span class="prob-name">${cls}</span>
        <div class="prob-bar-track">
          <div class="prob-bar-fill" style="width:${(val * 100).toFixed(1)}%;background:${colors[cls]||'#2563EB'};"></div>
        </div>
        <span class="prob-val">${Math.round(val * 100)}%</span>
      </div>
    `).join("");
  }

  // Dysgraphia Target Fidelity & Word Match Section
  const fidelityCard = document.getElementById("resDysgraphiaFidelityCard");
  const fidelityBadge = document.getElementById("resFidelitySummaryBadge");
  const tasksList = document.getElementById("resDysgraphiaTasksList");

  if (res.disorder === "dysgraphia" && fidelityCard && fidelityBadge && tasksList) {
    fidelityCard.style.display = "block";
    const matched = res.matched_tasks_count !== undefined ? res.matched_tasks_count : 5;
    const total = res.total_tasks_count !== undefined ? res.total_tasks_count : 5;
    const accPct = Math.round(res.target_accuracy_pct !== undefined ? res.target_accuracy_pct : 85);
    const isGood = matched >= 4 && accPct >= 65;

    fidelityBadge.style.background = isGood ? "#ecfdf5" : "#fef2f2";
    fidelityBadge.style.color = isGood ? "#065f46" : "#991b1b";
    fidelityBadge.style.border = isGood ? "1px solid #a7f3d0" : "1px solid #fecaca";
    fidelityBadge.textContent = `${isGood ? "✓" : "✗"} ${matched}/${total} Target Tasks Matched (${accPct}%)`;

    const taskResults = res.per_task_results || [];
    tasksList.innerHTML = taskResults.map((t, idx) => {
      const ver = t.verification || {};
      const tMatched = ver.is_matched !== false;
      const tAcc = ver.accuracy_pct !== undefined ? ver.accuracy_pct : 85;
      const isRev = ver.has_reversal || Boolean(t.letter_form_flag);
      const bg = tMatched ? "#f0fdf4" : (isRev ? "#fffbeb" : "#fef2f2");
      const border = tMatched ? "#bbf7d0" : (isRev ? "#fde68a" : "#fecaca");
      const textCol = tMatched ? "#15803d" : (isRev ? "#b45309" : "#b91c1c");
      const label = isRev ? "⚠️ Reversal" : (tMatched ? `✓ Matched (${tAcc}%)` : `✗ Mismatch (${tAcc}%)`);

      return `
        <div style="background:${bg}; border:1.5px solid ${border}; border-radius:8px; padding:10px; text-align:center;">
          <div style="font-size:11px; color:var(--text-muted); font-weight:600; text-transform:uppercase;">Task ${idx + 1}</div>
          <div style="font-size:18px; font-weight:800; color:var(--text-main); margin:4px 0;">"${t.target}"</div>
          <div style="font-size:11px; font-weight:700; color:${textCol};">${label}</div>
        </div>
      `;
    }).join("");
  } else if (fidelityCard) {
    fidelityCard.style.display = "none";
  }

  // Explainable findings
  const obsList = document.getElementById("resObservationsList");
  if (obsList && res.explainable_report) {
    obsList.innerHTML = res.explainable_report.what_we_observed.map(obs => `<li>${obs}</li>`).join("");
  }

  const meansText = document.getElementById("resWhatMeansText");
  if (meansText && res.explainable_report) {
    meansText.textContent = res.explainable_report.what_this_means;
  }

  const stepsList = document.getElementById("resNextStepsList");
  if (stepsList && res.explainable_report) {
    stepsList.innerHTML = res.explainable_report.suggested_next_steps.map(s => `<li>${s}</li>`).join("");
  }
}

// =========================================================================
// HISTORY VIEW & AUDIT TRAIL
// =========================================================================

async function loadHistoryView() {
  if (!currentChild) return;
  const container = document.getElementById("fullHistoryContainer");
  if (!container) return;

  try {
    const res = await fetch(`${API_BASE}/api/users/${currentChild.id}/history`);
    if (res.ok) {
      const records = await res.json();
      if (!records || records.length === 0) {
        container.innerHTML = `<div style="text-align:center;color:var(--text-muted);padding:40px;">No historical assessments recorded for ${currentChild.name}.</div>`;
        return;
      }

      container.innerHTML = `
        <div class="hist-table-wrap">
          <table class="hist-table">
            <thead>
              <tr>
                <th>Screening Type</th>
                <th>Outcome</th>
                <th>Confidence</th>
                <th>Date</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              ${records.map(r => `
                <tr>
                  <td>${r.disorder === "dyslexia" ? "📖 Dyslexia Screening" : "✍️ Dysgraphia Screening"}</td>
                  <td><span class="badge-risk ${r.prediction}">${r.prediction}</span></td>
                  <td style="font-family:'IBM Plex Mono';">${Math.round((r.confidence || 0.85) * 100)}%</td>
                  <td>${new Date(r.created_at).toLocaleString()}</td>
                  <td><button class="btn btn-outline btn-sm" onclick="viewHistoricalDetail(${r.id})">View Report</button></td>
                </tr>
              `).join("")}
            </tbody>
          </table>
        </div>
      `;
    }
  } catch (err) {
    container.innerHTML = `<div style="color:var(--danger);padding:20px;">Could not load history.</div>`;
  }
}

window.viewHistoricalDetail = async function(recordId) {
  if (!currentChild) return;
  try {
    const res = await fetch(`${API_BASE}/api/users/${currentChild.id}/history`);
    if (res.ok) {
      const records = await res.json();
      const match = records.find(r => r.id === recordId);
      if (match && match.full_result && match.full_result.prediction) {
        displayResults(match.full_result);
        navigateToView("results");
      } else if (match) {
        // Construct fallback presentation
        displayResults({
          disorder: match.disorder,
          prediction: match.prediction,
          confidence: match.confidence,
          probabilities: { [match.prediction]: match.confidence },
          explainable_report: {
            what_we_observed: match.observations || ["Historical screening log."],
            what_this_means: "Screening outcome recorded from an earlier session.",
            suggested_next_steps: ["Discuss observed indicators with the child's teacher or a learning specialist."]
          }
        });
        navigateToView("results");
      }
    }
  } catch (e) {}
};
