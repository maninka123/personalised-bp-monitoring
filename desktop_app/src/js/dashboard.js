/* ── dashboard.js — Clinical consultation dashboard ─────────────── */

let activeCharts = [];
let currentReadings = null;
let currentProfile = null;
let _resizeObserver = null;

const CARD_HINTS = {
  "24h Average BP": "The average of all blood pressure readings over the full 24-hour recording period.",
  "Awake BP": "Average blood pressure during daytime / awake hours only.",
  "Sleep BP": "Average blood pressure during night-time / sleep hours only.",
  "Dipping Status": "How much blood pressure drops during sleep. A normal drop is 10–20%.",
  "Morning Surge": "How much blood pressure rises in the first 2 hours after waking up.",
  "BP Variability": "How much readings change throughout the day. High variability may need review.",
  "Priority": "Suggested review urgency based on the detected patterns.",
  "Valid Readings": "Number of usable readings from the ABPM recording.",
};

const SECTION_HINTS = {
  "data-preview":  "First few rows of the uploaded blood pressure readings.",
  "24h-chart":     "Blood pressure plotted over 24 hours. Blue dots = awake, purple dots = asleep.",
  "awake-sleep":   "Average blood pressure compared between awake and sleep periods.",
  "profile-pos":   "Shows where this patient falls on the dipping / surge map.",
  "pattern-status": "Combined view of all detected patterns with clinical explanations.",
  "review-points": "Top review areas for the doctor based on the analysis findings.",
  "data-quality":  "Reliability indicators for this ABPM recording.",
  "day-night":     "Shows the expected vs observed blood pressure fall during sleep.",
};

function destroyCharts() {
  if (_resizeObserver) { _resizeObserver.disconnect(); _resizeObserver = null; }
  activeCharts.forEach(c => { try { c.destroy(); } catch {} });
  activeCharts = [];
}

function levelClass(level) {
  return { green: "level-green", yellow: "level-yellow", red: "level-red", grey: "level-grey" }[level] || "level-grey";
}
function bpValue(s, d) { return (s == null || d == null || isNaN(s) || isNaN(d)) ? "N/A" : `${Math.round(s)}/${Math.round(d)}`; }
function prettyCategory(v) { return v ? v.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase()) : "N/A"; }
function sleepLevel(p) { return (p.sleep_valid_readings || 0) < 3 ? "grey" : p.hypertensive_sleep ? "yellow" : "green"; }
function dippingLevel(p) { const c = p.dipping_category; return c === "insufficient_sleep" ? "grey" : c === "reverse_dipper" ? "red" : (c === "non_dipper" || c === "extreme_dipper") ? "yellow" : "green"; }
function morningLevel(p) { return (p.morning_surge_sbp == null || isNaN(p.morning_surge_sbp)) ? "grey" : p.morning_surge_high ? "yellow" : "green"; }
function priorityLevel(pri) { return pri === "High review priority" ? "red" : (pri === "Review soon" || pri === "Data review needed") ? "yellow" : "green"; }

function infoBtn(hint) {
  return `<button class="info-btn" onclick="showInfoTooltip(event, '${hint.replace(/'/g, "\\'")}')">i</button>`;
}
function sectionHeader(title, hintKey) {
  const hint = SECTION_HINTS[hintKey] || CARD_HINTS[hintKey] || "";
  return `<h3>${title} ${hint ? infoBtn(hint) : ""}</h3>`;
}

/* ═══════════════════════════════════════════════════════════════════
   MAIN RENDER — Clinical Consultation Layout
   ═══════════════════════════════════════════════════════════════════ */
function renderDashboard(container, profile) {
  destroyCharts();
  container.innerHTML = "";
  currentReadings = profile.readings || [];
  currentProfile = profile;

  // ── 1. Toolbar ──
  container.appendChild(buildToolbar());

  // ── 2. Traffic-light summary banner ──
  container.appendChild(buildTrafficBanner(profile));

  // ── 3. Summary cards (compact) ──
  container.appendChild(buildSummaryCards(profile));

  // ── 4. 24h BP curve with caption ──
  container.appendChild(build24hSection(profile));

  // ── 5. Day-vs-night dipping visual + Awake/Sleep bar chart ──
  container.appendChild(buildDipAndBarRow(profile));

  // ── 6. Profile Position chart ──
  container.appendChild(buildProfilePositionSection(profile));

  // ── 7. Clinical status table (replaces pattern flags) ──
  container.appendChild(buildClinicalStatusSection(profile));

  // ── 8. Patient-friendly summary card ──
  container.appendChild(buildPatientSummary(profile));

  // ── 9. Data quality box ──
  container.appendChild(buildDataQualityBox(profile));

  // ── 10. Data preview ──
  if (profile.readings && profile.readings.length) {
    container.appendChild(buildDataPreview(profile));
  }

  // ── 11. Disclaimer ──
  container.appendChild(buildDisclaimer());

  if (typeof refreshAssistantSummary === "function") {
    refreshAssistantSummary();
  }

  // ── ResizeObserver ──
  _resizeObserver = new ResizeObserver(() => {
    activeCharts.forEach(c => { try { c.resize(); } catch {} });
  });
  _resizeObserver.observe(container);
}

/* ── Toolbar ───────────────────────────────────────────────────── */
function buildToolbar() {
  const d = document.createElement("div");
  d.className = "dashboard-toolbar";
  d.innerHTML = `
    <button class="btn-report" id="btn-save-report" onclick="saveReport()">
      <svg width="16" height="16" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M6 2a2 2 0 00-2 2v12a2 2 0 002 2h8a2 2 0 002-2V7.414A2 2 0 0015.414 6L12 2.586A2 2 0 0010.586 2H6zm5 6a1 1 0 10-2 0v3.586l-1.293-1.293a1 1 0 10-1.414 1.414l3 3a1 1 0 001.414 0l3-3a1 1 0 00-1.414-1.414L11 11.586V8z" clip-rule="evenodd"/></svg>
      Save Report
    </button>
    <button class="btn-report btn-report-secondary" id="btn-open-assistant" onclick="switchView('ask-report'); refreshAssistantSummary();">
      <svg width="16" height="16" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M18 10c0 3.866-3.582 7-8 7a9.1 9.1 0 01-2.347-.306l-3.36 1.12a1 1 0 01-1.265-1.265l1.12-3.36A6.57 6.57 0 012 10c0-3.866 3.582-7 8-7s8 3.134 8 7zM7 9a1 1 0 100 2h6a1 1 0 100-2H7z" clip-rule="evenodd"/></svg>
      Ask About This BP Report
    </button>
    <span style="color:#94a3b8;font-size:0.82rem;">Generate a PDF report with charts and findings</span>`;
  return d;
}

/* ── 1. Traffic-light summary banner ───────────────────────────── */
function buildTrafficBanner(p) {
  const level = p.priority_level || priorityLevel(p.priority || "");
  const label = p.profile_label || prettyCategory(p.dipping_category);
  const caption = p.curve_caption || "";
  const priorityEmoji = { green: "✅", yellow: "⚠️", red: "🔴", grey: "⚪" }[level] || "";
  const details = p.patient_details || {};
  const detailBits = [
    details["Patient Name"] ? `Name: ${details["Patient Name"]}` : "",
    details["Patient ID"] ? `ID: ${details["Patient ID"]}` : "",
    details["Age"] ? `Age: ${details["Age"]}` : "",
    details["Sex"] ? `Sex: ${details["Sex"]}` : "",
    details["BMI"] ? `BMI: ${details["BMI"]}` : "",
  ].filter(Boolean).join(" | ");

  const d = document.createElement("div");
  d.className = "traffic-banner";
  d.innerHTML = `
    <div class="traffic-light ${level}"></div>
    <div class="traffic-body">
      <span class="tb-profile">Overall BP Profile: ${label}</span>
      <span class="tb-priority ${level}">${priorityEmoji} ${p.priority || ""}</span>
      ${detailBits ? `<div class="tb-issue"><strong>Patient:</strong> ${detailBits}</div>` : ""}
      <div class="tb-issue"><strong>Main finding:</strong> ${caption}</div>
    </div>`;
  return d;
}

/* ── 2. Summary cards (compact row) ────────────────────────────── */
function buildSummaryCards(profile) {
  const cards = [
    { label: "24h Average BP", value: bpValue(profile.mean_24h_sbp, profile.mean_24h_dbp), detail: "mmHg", level: profile.hypertensive_24h ? "yellow" : "green" },
    { label: "Awake BP", value: bpValue(profile.awake_mean_sbp, profile.awake_mean_dbp), detail: "mmHg", level: profile.hypertensive_awake ? "yellow" : "green" },
    { label: "Sleep BP", value: bpValue(profile.sleep_mean_sbp, profile.sleep_mean_dbp), detail: "mmHg", level: sleepLevel(profile) },
    { label: "Morning Surge", value: (profile.morning_surge_sbp != null && !isNaN(profile.morning_surge_sbp)) ? Math.round(profile.morning_surge_sbp) + "" : "N/A", detail: "mmHg", level: morningLevel(profile) },
  ];

  const d = document.createElement("div");
  d.className = "cards-grid";
  d.style.gridTemplateColumns = "repeat(4, 1fr)";
  cards.forEach(card => {
    const hint = CARD_HINTS[card.label] || "";
    d.innerHTML += `
      <div class="bp-card ${levelClass(card.level)}">
        <div class="card-label-row"><span class="card-label">${card.label}</span>${hint ? infoBtn(hint) : ""}</div>
        <div class="card-value">${card.value}</div>
        <div class="card-detail">${card.detail}</div>
      </div>`;
  });
  return d;
}

/* ── 3. 24h BP curve with caption ──────────────────────────────── */
function build24hSection(profile) {
  const d = document.createElement("div");
  d.className = "charts-row";
  const caption = profile.curve_caption || "No major pattern detected.";
  const isOk = caption.startsWith("No major");
  d.innerHTML = `<div class="chart-card">
    ${sectionHeader("📈 24-Hour Blood Pressure Curve", "24h-chart")}
    <div class="chart-wrap" style="height:320px;"><canvas id="chart-24h"></canvas></div>
    <div class="curve-caption ${isOk ? "ok" : ""}">📋 ${caption}</div>
  </div>`;
  // Defer rendering to after DOM append
  setTimeout(() => {
    const c = render24hChart(document.getElementById("chart-24h"), profile);
    if (c) activeCharts.push(c);
  }, 50);
  return d;
}

/* ── 4. Day-vs-night dipping visual + awake/sleep bar ──────────── */
function buildDipAndBarRow(profile) {
  const d = document.createElement("div");
  d.className = "charts-row-2";

  // Left: Dipping visual
  const awakeSBP = profile.awake_mean_sbp != null ? Math.round(profile.awake_mean_sbp) : "—";
  const sleepSBP = profile.sleep_mean_sbp != null ? Math.round(profile.sleep_mean_sbp) : "—";
  const dipPct = profile.dipping_pct_sbp;
  const dipText = dipPct != null ? `${Math.abs(dipPct).toFixed(1)}% ${dipPct > 0 ? 'fall' : 'rise'}` : "N/A";
  const isNonDip = profile.dipping_category === "non_dipper" || profile.dipping_category === "reverse_dipper";

  d.innerHTML = `
    <div class="dipping-visual">
      ${sectionHeader("☀️🌙 Day vs Night BP", "day-night")}
      <div class="dip-flow">
        <div class="dip-box"><div class="dip-label">☀️ Awake SBP</div><div class="dip-val">${awakeSBP}</div></div>
        <span class="dip-arrow">→</span>
        <div class="dip-box"><div class="dip-label">🌙 Sleep SBP</div><div class="dip-val">${sleepSBP}</div></div>
      </div>
      <div class="dip-info">
        <span class="dip-expected">Expected: should fall by about 10–20%</span>
        <span class="dip-observed ${isNonDip ? 'warn' : ''}">Observed: ${dipText}</span>
      </div>
    </div>
    <div class="chart-card">
      ${sectionHeader("Awake vs Sleep BP", "awake-sleep")}
      <div class="chart-wrap" style="height:280px;"><canvas id="chart-awake-sleep"></canvas></div>
    </div>`;

  setTimeout(() => {
    const c = renderAwakeSleepChart(document.getElementById("chart-awake-sleep"), profile);
    if (c) activeCharts.push(c);
  }, 50);
  return d;
}

/* ── 5. Profile Position chart ─────────────────────────────────── */
function buildProfilePositionSection(profile) {
  const d = document.createElement("div");
  d.className = "charts-row";
  d.innerHTML = `<div class="chart-card">
    ${sectionHeader("📍 Profile Position", "profile-pos")}
    <div class="chart-wrap" style="height:340px;"><canvas id="chart-profile"></canvas></div>
    <p style="color:#94a3b8;font-size:0.78rem;margin-top:0.5rem;line-height:1.5;">
      <strong>Reading this chart:</strong> Left side means BP did not fall enough during sleep.
      Higher position means stronger BP rise after waking.
    </p>
  </div>`;
  setTimeout(() => {
    const c = renderProfilePositionChart(document.getElementById("chart-profile"), profile);
    if (c) activeCharts.push(c);
  }, 50);
  return d;
}

/* ── 6. Clinical status table ──────────────────────────────────── */
function buildClinicalStatusSection(profile) {
  const table = profile.clinical_status_table || [];
  let rowsHtml = table.map(r => {
    const dotClass = r.status === "Needs review" ? "review" : r.status === "Within range" ? "range" : "normal";
    return `<tr>
      <td style="font-weight:600;color:var(--text);">${r.pattern}</td>
      <td><span class="status-dot ${dotClass}">${r.status}</span></td>
      <td>${r.why}</td>
      <td>${r.review}</td>
    </tr>`;
  }).join("");

  const d = document.createElement("div");
  d.className = "data-table-wrap";
  d.innerHTML = `${sectionHeader("🩺 Pattern Status", "pattern-status")}
    <table class="clinical-table">
      <thead><tr><th>Pattern</th><th>Status</th><th>Why it matters</th><th>Review point</th></tr></thead>
      <tbody>${rowsHtml}</tbody>
    </table>`;
  return d;
}

/* ── 7. Patient-friendly summary card ──────────────────────────── */
function buildPatientSummary(profile) {
  const explanation = profile.patient_explanation || "";
  const reviews = (profile.review_points || []).map(r => r["Doctor review point"] || "");
  const reviewTags = reviews.map(r => `<li>${r}</li>`).join("");

  const d = document.createElement("div");
  d.className = "patient-summary-card";
  d.innerHTML = `
    <h3>💬 Your main BP pattern</h3>
    <div class="summary-text">${explanation}</div>
    ${reviewTags ? `<ul class="review-list">${reviewTags}</ul>` : ""}`;
  return d;
}

/* ── 8. Data quality box ───────────────────────────────────────── */
function buildDataQualityBox(profile) {
  const sleepSource = profile.sleep_source || "Entered sleep time";
  const d = document.createElement("div");
  d.className = "data-quality-box";
  d.innerHTML = `${sectionHeader("📋 Data Quality", "data-quality")}
    <div class="dq-grid">
      <div class="dq-item"><div class="dq-label">Valid readings</div><div class="dq-val">${profile.valid_readings || "—"}</div></div>
      <div class="dq-item"><div class="dq-label">Sleep readings</div><div class="dq-val">${profile.sleep_valid_readings || "—"}</div></div>
      <div class="dq-item"><div class="dq-label">Sleep/wake source</div><div class="dq-val" style="font-size:0.82rem;">${sleepSource}</div></div>
      <div class="dq-item"><div class="dq-label">Quality</div><div class="dq-val" style="font-size:0.82rem;">${profile.data_quality || "—"}</div></div>
    </div>`;
  return d;
}

/* ── 9. Data preview ───────────────────────────────────────────── */
function buildDataPreview(profile) {
  let rows = profile.readings.slice(0, 8);
  let rowsHtml = rows.map(r => `
    <tr>
      <td>${r.time_display || ""}</td>
      <td><strong>${r.Systolic}</strong></td>
      <td><strong>${r.Diastolic}</strong></td>
      <td>${r.HR != null ? r.HR : "—"}</td>
      <td><span class="badge ${r.Wake_Sleep === 0 ? "badge-sleep" : "badge-awake"}">${r.Wake_Sleep === 0 ? "🌙 Asleep" : "☀️ Awake"}</span></td>
    </tr>`).join("");

  const d = document.createElement("div");
  d.className = "data-table-wrap";
  d.innerHTML = `${sectionHeader("📊 Patient Readings Preview", "data-preview")}
    <table class="data-table">
      <thead><tr><th>Time</th><th>Top (SBP)</th><th>Bottom (DBP)</th><th>Pulse</th><th>State</th></tr></thead>
      <tbody>${rowsHtml}</tbody>
    </table>`;
  return d;
}

/* ── 10. Disclaimer ────────────────────────────────────────────── */
function buildDisclaimer() {
  const d = document.createElement("div");
  d.className = "warning-banner";
  d.innerHTML = `<svg viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clip-rule="evenodd"/></svg>
    This dashboard supports clinician review. It does <strong>not</strong> recommend automatic medication changes.`;
  return d;
}

/* ── Info tooltip system ───────────────────────────────────────── */
function showInfoTooltip(event, text) {
  event.stopPropagation();
  const existing = document.querySelector(".info-tooltip");
  if (existing) existing.remove();
  const btn = event.currentTarget;
  const rect = btn.getBoundingClientRect();
  const tip = document.createElement("div");
  tip.className = "info-tooltip";
  tip.textContent = text;
  tip.style.top = (rect.bottom + 8) + "px";
  tip.style.left = Math.max(8, rect.left - 20) + "px";
  document.body.appendChild(tip);
  setTimeout(() => {
    document.addEventListener("click", function handler() {
      tip.remove();
      document.removeEventListener("click", handler);
    });
  }, 50);
}
