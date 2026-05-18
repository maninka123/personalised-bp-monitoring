/* ── app.js — Main application controller ── */

let API_BASE = "http://127.0.0.1:18347";

function switchView(name) {
  document.querySelectorAll(".view").forEach(v => v.classList.remove("active"));
  document.querySelectorAll(".nav-btn").forEach(b => b.classList.remove("active"));
  const view = document.getElementById("view-" + name);
  const btn = document.querySelector(`[data-view="${name}"]`);
  if (view) view.classList.add("active");
  if (btn) btn.classList.add("active");
}

/* ── Toast notifications ───────────────────────────────────────── */
function showToast(message, type, path) {
  const container = document.getElementById("toast-container");
  if (!container) return;
  const icon = type === "success" ? "✅" : type === "error" ? "❌" : "ℹ️";
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  toast.innerHTML = `
    <span class="toast-icon">${icon}</span>
    <div class="toast-msg">${message}${path ? `<br><span class="toast-path">${path}</span>` : ""}</div>
    <button class="toast-close" onclick="this.parentElement.remove()">✕</button>`;
  container.appendChild(toast);
  setTimeout(() => { if (toast.parentElement) toast.remove(); }, 5000);
}

/* ── Report saving ─────────────────────────────────────────────── */
async function saveReport() {
  const btn = document.getElementById("btn-save-report");
  if (!btn || !currentReadings || currentReadings.length === 0) return;
  btn.disabled = true;
  btn.innerHTML = `<svg width="16" height="16" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-12a1 1 0 10-2 0v4a1 1 0 00.293.707l2.828 2.829a1 1 0 101.415-1.415L11 9.586V6z" clip-rule="evenodd"/></svg> Generating…`;

  try {
    // Build readings from current data (reconstruct Time + Wake_Sleep)
    const readings = currentReadings.map(r => ({
      Time: r.time_display ? r.time_display.split(" ").pop() + ":00" : "00:00:00",
      Systolic: r.Systolic,
      Diastolic: r.Diastolic,
      HR: r.HR,
      Wake_Sleep: r.Wake_Sleep,
    }));

    const resp = await fetch(`${API_BASE}/api/export-pdf`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        readings,
        patient_id: currentProfile?.patient_details?.["Patient ID"] || "PATIENT",
        patient_name: currentProfile?.patient_details?.["Patient Name"] || "",
        patient_age: currentProfile?.patient_details?.["Age"] || "",
        patient_sex: currentProfile?.patient_details?.["Sex"] || "",
        patient_bmi: currentProfile?.patient_details?.["BMI"] || "",
        abpm_date: currentProfile?.patient_details?.["ABPM date"] || new Date().toISOString().split("T")[0],
        sleep_start: "22:00",
        sleep_end: "07:00",
      }),
    });
    if (!resp.ok) throw new Error(await resp.text());
    const data = await resp.json();

    if (!window.api || !window.api.saveFile) {
      showToast("Save dialog not available in browser mode.", "error");
      return;
    }

    const defaultName = `bp_report_${new Date().toISOString().slice(0,10)}.pdf`;
    const savedPath = await window.api.saveFile(data.pdf_base64, defaultName);

    if (savedPath) {
      showToast("Report saved successfully!", "success", savedPath);
    }
  } catch (err) {
    showToast("Failed to generate report: " + err.message, "error");
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<svg width="16" height="16" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M6 2a2 2 0 00-2 2v12a2 2 0 002 2h8a2 2 0 002-2V7.414A2 2 0 0015.414 6L12 2.586A2 2 0 0010.586 2H6zm5 6a1 1 0 10-2 0v3.586l-1.293-1.293a1 1 0 10-1.414 1.414l3 3a1 1 0 001.414 0l3-3a1 1 0 00-1.414-1.414L11 11.586V8z" clip-rule="evenodd"/></svg> Save Report`;
  }
}

/* ── Init ───────────────────────────────────────────────────────── */
async function initApp() {
  try { await loadChartJs(); } catch(e) { console.warn("Chart.js load failed:", e); }

  if (window.api && window.api.getApiBase) {
    API_BASE = await window.api.getApiBase();
  }

  // Nav buttons
  document.querySelectorAll(".nav-btn").forEach(btn => {
    btn.addEventListener("click", () => switchView(btn.dataset.view));
  });

  // Init manual entry
  initManualEntry(document.getElementById("manual-entry-container"));
  initReportAssistant();

  // File upload
  const chooseBtn = document.getElementById("btn-choose-file");
  if (chooseBtn) {
    chooseBtn.addEventListener("click", handleFileUpload);
    document.getElementById("upload-box").addEventListener("click", (e) => {
      if (e.target.closest("button")) return;
      handleFileUpload();
    });
  }
  const samplesBtn = document.getElementById("btn-open-samples");
  if (samplesBtn) {
    samplesBtn.addEventListener("click", openSampleInputs);
  }

  loadExamplePatient();
}

/* Ask About This BP Report */
function initReportAssistant() {
  const askBtn = document.getElementById("btn-ask-report");
  const questionBox = document.getElementById("assistant-question");

  document.querySelectorAll(".quick-question-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const question = btn.dataset.question || btn.textContent;
      if (questionBox) questionBox.value = question;
      askCurrentReport(question);
    });
  });

  if (askBtn) {
    askBtn.addEventListener("click", () => {
      const question = questionBox ? questionBox.value : "";
      askCurrentReport(question);
    });
  }

  if (questionBox) {
    questionBox.addEventListener("keydown", event => {
      if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
        askCurrentReport(questionBox.value);
      }
    });
  }

  refreshAssistantStatus();
  refreshAssistantSummary();
}

async function refreshAssistantStatus() {
  const statusEl = document.getElementById("assistant-status-text");
  if (!statusEl) return;
  try {
    const resp = await fetch(`${API_BASE}/api/assistant-status`);
    if (!resp.ok) throw new Error("assistant status unavailable");
    const data = await resp.json();
    statusEl.textContent = `${data.model || "Gemma"} uses the saved local token automatically. No raw ABPM rows are sent.`;
  } catch {
    statusEl.textContent = "Gemma will be checked when the backend is ready.";
  }
}

function refreshAssistantSummary() {
  const box = document.getElementById("assistant-summary-box");
  if (!box) return;
  if (!currentProfile) {
    box.textContent = "Analyse a patient first, then ask about the calculated report.";
    return;
  }
  const profile = currentProfile.profile_label || "Current BP profile";
  const awake = bpText(currentProfile.awake_mean_sbp, currentProfile.awake_mean_dbp);
  const sleep = bpText(currentProfile.sleep_mean_sbp, currentProfile.sleep_mean_dbp);
  const dipping = valueText(currentProfile.dipping_pct_sbp, "%", 1);
  const surge = valueText(currentProfile.morning_surge_sbp, " mmHg", 0);
  const variability = currentProfile.high_variability ? "High" : "Not flagged";
  const priority = currentProfile.priority || "Not assigned";
  box.innerHTML = `
    <div class="assistant-summary-grid">
      <div><span>Profile</span><strong>${escapeHtml(profile)}</strong></div>
      <div><span>Priority</span><strong>${escapeHtml(priority)}</strong></div>
      <div><span>Awake BP</span><strong>${awake}</strong></div>
      <div><span>Sleep BP</span><strong>${sleep}</strong></div>
      <div><span>Dipping</span><strong>${dipping}</strong></div>
      <div><span>Morning surge</span><strong>${surge}</strong></div>
      <div><span>Variability</span><strong>${variability}</strong></div>
      <div><span>Data quality</span><strong>${escapeHtml(currentProfile.data_quality || "N/A")}</strong></div>
    </div>`;
}

async function askCurrentReport(question) {
  const cleanQuestion = (question || "").trim();
  const answerEl = document.getElementById("assistant-answer");
  const askBtn = document.getElementById("btn-ask-report");

  if (!answerEl) return;
  if (!currentProfile) {
    answerEl.textContent = "Load or analyse a patient first, then ask about the calculated BP report.";
    showToast("Analyse a patient first.", "error");
    return;
  }
  if (!cleanQuestion) {
    answerEl.textContent = "Type a question about the current BP report.";
    return;
  }

  refreshAssistantSummary();
  answerEl.innerHTML = `<div class="assistant-loading">Asking Gemma about the report summary...</div>`;
  if (askBtn) askBtn.disabled = true;

  try {
    const resp = await fetch(`${API_BASE}/api/ask-report`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ profile: assistantProfileSummary(currentProfile), question: cleanQuestion }),
    });
    if (!resp.ok) throw new Error(await responseError(resp));
    const data = await resp.json();
    answerEl.innerHTML = "";
    const questionNode = document.createElement("div");
    questionNode.className = "assistant-question-chip";
    questionNode.textContent = cleanQuestion;
    const answerNode = document.createElement("div");
    answerNode.className = "assistant-answer-text";
    answerNode.innerHTML = formatAssistantAnswer(data.answer || "No answer returned.");
    const footNode = document.createElement("div");
    footNode.className = "assistant-answer-source";
    footNode.textContent = "Source: Gemma using calculated report summary only";
    answerEl.appendChild(questionNode);
    answerEl.appendChild(answerNode);
    answerEl.appendChild(footNode);
  } catch (err) {
    answerEl.textContent = "Gemma could not answer right now: " + err.message;
  } finally {
    if (askBtn) askBtn.disabled = false;
  }
}

async function responseError(resp) {
  try {
    const data = await resp.json();
    return data.detail || JSON.stringify(data);
  } catch {
    return resp.text();
  }
}

function bpText(sbp, dbp) {
  if (sbp == null || dbp == null || isNaN(sbp) || isNaN(dbp)) return "N/A";
  return `${Math.round(sbp)}/${Math.round(dbp)}`;
}

function valueText(value, unit, decimals) {
  if (value == null || isNaN(value)) return "N/A";
  return `${Number(value).toFixed(decimals)}${unit}`;
}

function assistantProfileSummary(profile) {
  const {
    readings,
    clinical_status_table,
    pattern_flags,
    review_points,
    ...summary
  } = profile || {};
  return {
    ...summary,
    clinical_status_table: clinical_status_table || [],
    pattern_flags: pattern_flags || [],
    review_points: review_points || [],
  };
}

function formatAssistantAnswer(text) {
  const lines = String(text || "").replace(/\r\n/g, "\n").split("\n");
  const html = [];
  let listOpen = false;

  function closeList() {
    if (listOpen) {
      html.push("</ul>");
      listOpen = false;
    }
  }

  lines.forEach(line => {
    const trimmed = line.trim();
    if (!trimmed) {
      closeList();
      return;
    }

    const bullet = trimmed.match(/^[-*]\s+(.+)$/);
    if (bullet) {
      if (!listOpen) {
        html.push("<ul>");
        listOpen = true;
      }
      html.push(`<li>${inlineAssistantMarkdown(bullet[1])}</li>`);
      return;
    }

    closeList();
    if (/^\*\*[^*]+:\*\*$/.test(trimmed) || /^\*\*[^*]+\*\*$/.test(trimmed)) {
      html.push(`<h4>${inlineAssistantMarkdown(trimmed)}</h4>`);
    } else {
      html.push(`<p>${inlineAssistantMarkdown(trimmed)}</p>`);
    }
  });

  closeList();
  return html.join("");
}

function inlineAssistantMarkdown(text) {
  return escapeHtml(text)
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\*([^*]+)\*/g, "<em>$1</em>");
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

async function loadExamplePatient() {
  try {
    const resp = await fetch(`${API_BASE}/api/analyse`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        readings: exampleReadings(),
        patient_id: "EXAMPLE",
        sleep_start: "22:00",
        sleep_end: "07:00",
      }),
    });
    if (!resp.ok) throw new Error("API not ready");
    const profile = await resp.json();
    renderDashboard(document.getElementById("dashboard-container"), profile);
    document.querySelector("#view-dashboard .view-subtitle").textContent = "Example patient data";
  } catch {
    setTimeout(loadExamplePatient, 2000);
  }
}

async function handleFileUpload() {
  if (!window.api || !window.api.openFileDialog) return;
  const file = await window.api.openFileDialog();
  if (!file) return;

  const patientId = document.getElementById("file-patient-id").value || "NEW";
  const sleepStart = document.getElementById("file-sleep-start").value || "22:00";
  const sleepEnd = document.getElementById("file-sleep-end").value || "07:00";

  try {
    const resp = await fetch(`${API_BASE}/api/analyse-file`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        filename: file.name,
        data_base64: file.buffer,
        patient_id: patientId,
        sleep_start: sleepStart,
        sleep_end: sleepEnd,
      }),
    });
    if (!resp.ok) throw new Error(await resp.text());
    const profile = await resp.json();
    switchView("dashboard");
    const loadedPatient = patientDisplayName(profile.patient_details, patientId);
    document.querySelector("#view-dashboard .view-subtitle").textContent = `${loadedPatient} (from ${file.name})`;
    renderDashboard(document.getElementById("dashboard-container"), profile);
    showToast("File loaded and analysed successfully!", "success");
  } catch(e) {
    showToast("File analysis failed: " + e.message, "error");
  }
}

async function openSampleInputs() {
  if (!window.api || !window.api.openSampleInputs) {
    showToast("Sample folder is only available in the desktop app.", "error");
    return;
  }
  try {
    const folderPath = await window.api.openSampleInputs();
    showToast("Sample patient files opened.", "success", folderPath);
  } catch (err) {
    showToast("Could not open sample files: " + err.message, "error");
  }
}

function patientDisplayName(details, fallbackId) {
  const patientName = details?.["Patient Name"];
  const patientId = details?.["Patient ID"] || fallbackId || "NEW";
  if (patientName) return `Patient: ${patientName} (${patientId})`;
  return `Patient: ${patientId}`;
}

document.addEventListener("DOMContentLoaded", initApp);
