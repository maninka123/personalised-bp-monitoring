/* limited-predict.js - Limited-input ABPM-TSL prediction tab */

const LIMITED_FIELDS = [
  "clinic_sbp", "clinic_dbp", "resting_hr",
  "morning_home_sbp", "morning_home_dbp", "evening_home_sbp", "evening_home_dbp",
  "home_3day_mean_sbp", "home_3day_mean_dbp", "home_7day_mean_sbp", "home_7day_mean_dbp",
  "home_bp_variability", "age", "sex", "bmi", "diabetes", "smoker",
  "previous_hypertension", "medication_status", "sleep_duration", "sleep_quality",
  "caffeine_cups", "alcohol_units", "stress_level",
];

const LIMITED_REQUIRED_FIELDS = ["clinic_sbp", "clinic_dbp", "age", "sex"];

function initLimitedPredict(container) {
  const root = container || document.getElementById("limited-predict-container");
  if (!root) return;

  root.innerHTML = `
    <div class="limited-layout">
      <div class="limited-form-panel">
        <div class="limited-intro">
          <div>
            <span class="limited-eyebrow">ABPM-TSL estimate</span>
            <h3>Start with the basics, then add what you know</h3>
            <p>Complete the required fields first. Optional details can stay blank, but the prediction confidence usually improves when more patient context is entered.</p>
          </div>
        </div>

        <div class="limited-workflow-strip">
          <div><strong>1. Required</strong><span>Clinic BP, age, and sex</span></div>
          <div><strong>2. Helpful</strong><span>Home BP and BMI</span></div>
          <div><strong>3. Extra detail</strong><span>History, sleep, and lifestyle</span></div>
        </div>

        <div class="limited-section">
          <div class="limited-section-head">
            <h3><span class="limited-step">1</span>Required patient details</h3>
            <span>Complete these before predicting</span>
          </div>
          <div class="form-row">
            ${limitedNumber("clinic_sbp", "Clinic systolic BP", "mmHg", "1", true)}
            ${limitedNumber("clinic_dbp", "Clinic diastolic BP", "mmHg", "1", true)}
            ${limitedNumber("age", "Age", "years", "1", true)}
            ${limitedSelect("sex", "Sex", [["", "Select"], ["female", "Female"], ["male", "Male"], ["other", "Other"]], true)}
          </div>
        </div>

        <div class="limited-section">
          <div class="limited-section-head">
            <h3><span class="limited-step">2</span>Optional clinic details</h3>
            <span>Add if available</span>
          </div>
          <div class="form-row">
            ${limitedNumber("resting_hr", "Resting pulse", "bpm")}
            ${limitedNumber("bmi", "BMI", "kg/m2", "0.1")}
          </div>
        </div>

        <div class="limited-section">
          <div class="limited-section-head">
            <h3><span class="limited-step">3</span>Optional home BP</h3>
            <span>Morning, evening, and repeated averages help most</span>
          </div>
          <div class="limited-field-group">
            <h4>Morning and evening readings</h4>
            <div class="form-row">
              ${limitedNumber("morning_home_sbp", "Morning home SBP", "mmHg")}
              ${limitedNumber("morning_home_dbp", "Morning home DBP", "mmHg")}
              ${limitedNumber("evening_home_sbp", "Evening home SBP", "mmHg")}
              ${limitedNumber("evening_home_dbp", "Evening home DBP", "mmHg")}
            </div>
          </div>
          <div class="limited-field-group">
            <h4>Home averages</h4>
            <div class="form-row">
              ${limitedNumber("home_3day_mean_sbp", "3-day SBP average", "mmHg")}
              ${limitedNumber("home_3day_mean_dbp", "3-day DBP average", "mmHg")}
              ${limitedNumber("home_7day_mean_sbp", "7-day SBP average", "mmHg")}
              ${limitedNumber("home_7day_mean_dbp", "7-day DBP average", "mmHg")}
              ${limitedNumber("home_bp_variability", "Home BP variability", "SD or spread")}
            </div>
          </div>
        </div>

        <div class="limited-section">
          <div class="limited-section-head">
            <h3><span class="limited-step">4</span>Optional medical history</h3>
            <span>Leave unknown values blank</span>
          </div>
          <div class="form-row">
            ${limitedSelect("diabetes", "Diabetes", [["", "Unknown"], ["no", "No"], ["yes", "Yes"]])}
            ${limitedSelect("smoker", "Smoking", [["", "Unknown"], ["no", "Never"], ["former", "Former"], ["current", "Current"]])}
            ${limitedSelect("previous_hypertension", "Previous hypertension", [["", "Unknown"], ["no", "No"], ["yes", "Yes"]])}
            ${limitedSelect("medication_status", "Medication", [["", "Unknown"], ["untreated", "No BP medication"], ["treated", "On BP medication"]])}
          </div>
        </div>

        <div class="limited-section">
          <div class="limited-section-head">
            <h3><span class="limited-step">5</span>Optional lifestyle and sleep</h3>
            <span>Useful but not required</span>
          </div>
          <div class="form-row">
            ${limitedNumber("sleep_duration", "Sleep duration", "hours", "0.1")}
            ${limitedSelect("sleep_quality", "Sleep quality", [["", "Unknown"], ["good", "Good"], ["fair", "Fair"], ["poor", "Poor"]])}
            ${limitedNumber("caffeine_cups", "Caffeine", "cups/day", "0.5")}
            ${limitedNumber("alcohol_units", "Alcohol", "units/week", "1")}
            ${limitedSelect("stress_level", "Stress", [["", "Unknown"], ["1", "Very low"], ["2", "Low"], ["3", "Moderate"], ["4", "High"], ["5", "Very high"]])}
          </div>
        </div>

        <div class="limited-actions">
          <button class="btn btn-primary" id="limited-predict-btn">Predict ABPM Risk</button>
          <button class="btn btn-secondary" id="limited-clear-btn" type="button">Clear optional inputs</button>
        </div>
      </div>

        <div class="limited-results-panel" id="limited-results">
          <div class="limited-empty">
            <h3>Ready for limited-input prediction</h3>
            <p>Fill the required fields, add any optional details you have, then run the ABPM risk estimate.</p>
          </div>
        </div>
      </div>`;

  document.getElementById("limited-clear-btn").addEventListener("click", clearLimitedOptionalInputs);
  document.getElementById("limited-predict-btn").addEventListener("click", predictLimitedInputRisk);
}

function limitedNumber(id, label, suffix, step = "1", required = false) {
  const requiredAttr = required ? " required" : "";
  const requiredMark = required ? '<span class="limited-required-mark">Required</span>' : "";
  const placeholder = required ? "Required" : "Optional";
  return `<div class="form-group">
    <label for="limited-${id}">${label}${requiredMark}</label>
    <div class="limited-input-with-unit">
      <input type="number" id="limited-${id}" class="form-input" step="${step}" placeholder="${placeholder}"${requiredAttr}>
      <span>${suffix}</span>
    </div>
  </div>`;
}

function limitedSelect(id, label, options, required = false) {
  const optionHtml = options.map(([value, text]) => `<option value="${value}">${text}</option>`).join("");
  const requiredAttr = required ? " required" : "";
  const requiredMark = required ? '<span class="limited-required-mark">Required</span>' : "";
  return `<div class="form-group">
    <label for="limited-${id}">${label}${requiredMark}</label>
    <select id="limited-${id}" class="form-input"${requiredAttr}>${optionHtml}</select>
  </div>`;
}

function setLimitedValue(field, value) {
  const el = document.getElementById(`limited-${field}`);
  if (el) el.value = value == null ? "" : String(value);
}

function clearLimitedOptionalInputs() {
  [
    "resting_hr", "bmi",
    "morning_home_sbp", "morning_home_dbp", "evening_home_sbp", "evening_home_dbp",
    "home_3day_mean_sbp", "home_3day_mean_dbp", "home_7day_mean_sbp", "home_7day_mean_dbp",
    "home_bp_variability", "diabetes", "smoker", "previous_hypertension", "medication_status",
    "sleep_duration", "sleep_quality", "caffeine_cups", "alcohol_units", "stress_level",
  ].forEach(field => setLimitedValue(field, ""));
}

function buildLimitedPayload() {
  const payload = {
    model_variant: "distilled_ssl_student",
    input_scenario: "all_limited_inputs",
  };
  LIMITED_FIELDS.forEach(field => {
    const el = document.getElementById(`limited-${field}`);
    payload[field] = el ? el.value : "";
  });
  return payload;
}

function validateLimitedRequiredInputs() {
  const missing = LIMITED_REQUIRED_FIELDS
    .map(field => document.getElementById(`limited-${field}`))
    .find(el => !el || !String(el.value || "").trim());
  if (!missing) return true;

  const resultEl = document.getElementById("limited-results");
  if (resultEl) {
    resultEl.innerHTML = `<div class="limited-error">Please complete the required patient details before running the prediction.</div>`;
  }
  if (typeof showToast === "function") showToast("Complete the required fields first.", "error");
  missing.focus();
  return false;
}

async function predictLimitedInputRisk() {
  const btn = document.getElementById("limited-predict-btn");
  const resultEl = document.getElementById("limited-results");
  if (!resultEl) return;
  if (!validateLimitedRequiredInputs()) return;
  btn.disabled = true;
  btn.textContent = "Predicting...";
  resultEl.innerHTML = `<div class="limited-loading">Running missingness-aware risk estimate...</div>`;
  try {
    const resp = await fetch(`${API_BASE}/api/limited-input-predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(buildLimitedPayload()),
    });
    if (!resp.ok) throw new Error(await responseError(resp));
    const data = await resp.json();
    renderLimitedResult(data);
  } catch (err) {
    resultEl.innerHTML = `<div class="limited-error">Prediction failed: ${limitedEscape(err.message)}</div>`;
    if (typeof showToast === "function") showToast("Limited-input prediction failed: " + err.message, "error");
  } finally {
    btn.disabled = false;
    btn.textContent = "Predict ABPM Risk";
  }
}

function renderLimitedResult(data) {
  const resultEl = document.getElementById("limited-results");
  const risks = data.risk_probabilities || {};
  const priority = data.priority || {};
  const confidence = data.confidence || {};
  const targets = data.estimated_targets || {};
  const missing = data.missingness || {};
  resultEl.innerHTML = `
    <div class="limited-result-head">
      <div>
        <span class="limited-eyebrow">Risk estimate</span>
        <h3>${limitedEscape(priority.label || "Risk estimate")}</h3>
        <p>${limitedEscape(data.model || "")}</p>
      </div>
      <span class="limited-priority ${limitedEscape(priority.level || "grey")}">${Math.round((priority.score || 0) * 100)} priority score</span>
    </div>

    <div class="limited-confidence-row">
      <div><span>Confidence</span><strong>${limitedEscape(confidence.label || "N/A")} (${confidence.score || "N/A"}%)</strong></div>
      <div><span>Input groups</span><strong>${limitedEscape((confidence.present_groups || []).join(", ") || "None")}</strong></div>
      <div><span>Missing fields</span><strong>${missing.missing_count || 0}</strong></div>
    </div>

    <div class="limited-risk-list">
      ${limitedRiskBar("Abnormal dipping", risks.abnormal_dipping)}
      ${limitedRiskBar("Morning surge", risks.morning_surge_high)}
      ${limitedRiskBar("Nocturnal hypertension", risks.nocturnal_hypertension)}
      ${limitedRiskBar("High BP burden", risks.high_bp_burden)}
      ${limitedRiskBar("High variability", risks.high_variability)}
    </div>

    <div class="limited-target-grid">
      <div><span>Dipping proxy</span><strong>${targetText(targets.dipping_percentage_proxy, "%")}</strong></div>
      <div><span>Sleep SBP proxy</span><strong>${targetText(targets.sleep_mean_sbp_proxy, " mmHg")}</strong></div>
      <div><span>Morning surge proxy</span><strong>${targetText(targets.morning_surge_mmhg_proxy, " mmHg")}</strong></div>
      <div><span>BP burden score</span><strong>${targetText(targets.bp_burden_score, "/100")}</strong></div>
    </div>

    <div class="limited-text-block">
      <h4>Main drivers</h4>
      <ul>${(data.explanation || []).map(item => `<li>${limitedEscape(item)}</li>`).join("")}</ul>
    </div>

    <div class="limited-text-block">
      <h4>Recommended next data</h4>
      <ul>${(data.recommendations || []).map(item => `<li>${limitedEscape(item)}</li>`).join("")}</ul>
    </div>

    <div class="limited-missing-box">
      <h4>Missingness mask</h4>
      <p>${limitedEscape((missing.missing_features || []).slice(0, 14).join(", ") || "No missing fields in this limited-input form.")}</p>
    </div>

    <div class="limited-boundary">${limitedEscape(data.clinical_boundary || data.model_status || "")}</div>`;
}

function limitedRiskBar(label, value) {
  const pct = Math.round((Number(value || 0)) * 100);
  const level = pct >= 65 ? "red" : pct >= 45 ? "yellow" : "green";
  return `<div class="limited-risk-row">
    <div class="limited-risk-label"><span>${limitedEscape(label)}</span><strong>${pct}%</strong></div>
    <div class="limited-risk-track"><div class="limited-risk-fill ${level}" style="width:${pct}%"></div></div>
  </div>`;
}

function targetText(value, unit) {
  if (value == null || Number.isNaN(Number(value))) return "N/A";
  return `${Number(value).toFixed(1)}${unit}`;
}

function limitedEscape(value) {
  return String(value == null ? "" : value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
