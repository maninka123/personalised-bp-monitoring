/* ── manual-entry.js — Manual BP reading entry form ── */
let manualReadings = [];

function initManualEntry(container) {
  container.innerHTML = `
    <div class="entry-form-card">
      <h3>Add a Blood Pressure Reading</h3>
      <div class="form-row">
        <div class="form-group"><label for="me-time">Time</label><input type="time" id="me-time" value="08:00" class="form-input"></div>
        <div class="form-group"><label for="me-sbp">Top number (SBP)</label><input type="number" id="me-sbp" value="130" min="60" max="260" class="form-input"></div>
        <div class="form-group"><label for="me-dbp">Bottom number (DBP)</label><input type="number" id="me-dbp" value="80" min="30" max="160" class="form-input"></div>
        <div class="form-group"><label for="me-hr">Pulse</label><input type="number" id="me-hr" value="72" min="30" max="200" class="form-input"></div>
        <div class="form-group"><label for="me-state">State</label><select id="me-state" class="form-input"><option value="1">Awake</option><option value="0">Asleep</option></select></div>
      </div>
      <div class="readings-actions">
        <button class="btn btn-primary" id="me-add-btn">Add Reading</button>
        <button class="btn btn-secondary" id="me-example-btn">Load 18 Examples</button>
        <button class="btn btn-danger" id="me-clear-btn">Clear All</button>
      </div>
    </div>
    <div class="entry-form-card">
      <h3>Patient Details</h3>
      <div class="form-row">
        <div class="form-group"><label for="me-patient-id">Patient ID</label><input type="text" id="me-patient-id" value="MANUAL" class="form-input"></div>
        <div class="form-group"><label for="me-sleep-start">Usual sleep time</label><input type="time" id="me-sleep-start" value="22:00" class="form-input"></div>
        <div class="form-group"><label for="me-sleep-end">Usual wake time</label><input type="time" id="me-sleep-end" value="07:00" class="form-input"></div>
      </div>
    </div>
    <p class="reading-count" id="me-count"></p>
    <button class="btn btn-success" id="me-analyse-btn" disabled>Analyse Readings</button>
    <div class="data-table-wrap" style="margin-top:1rem;">
      <h3>Entered Readings</h3>
      <table class="data-table"><thead><tr><th>#</th><th>Time</th><th>SBP</th><th>DBP</th><th>Pulse</th><th>State</th><th></th></tr></thead><tbody id="me-tbody"></tbody></table>
    </div>`;

  document.getElementById("me-add-btn").onclick = () => {
    const t = document.getElementById("me-time").value;
    const s = parseInt(document.getElementById("me-sbp").value);
    const d = parseInt(document.getElementById("me-dbp").value);
    const h = parseInt(document.getElementById("me-hr").value);
    const w = parseInt(document.getElementById("me-state").value);
    if (!t||isNaN(s)||isNaN(d)) return;
    manualReadings.push({Time:t+":00",Systolic:s,Diastolic:d,HR:h,Wake_Sleep:w});
    refreshReadingsTable();
  };
  document.getElementById("me-clear-btn").onclick = () => { manualReadings=[]; refreshReadingsTable(); };
  document.getElementById("me-example-btn").onclick = () => { manualReadings=exampleReadings(); refreshReadingsTable(); };
  document.getElementById("me-analyse-btn").onclick = () => analyseManualReadings();
  refreshReadingsTable();
}

function refreshReadingsTable() {
  const tbody = document.getElementById("me-tbody");
  const countEl = document.getElementById("me-count");
  const btn = document.getElementById("me-analyse-btn");
  if (!tbody) return;
  tbody.innerHTML = manualReadings.map((r,i) =>
    `<tr><td>${i+1}</td><td>${r.Time}</td><td>${r.Systolic}</td><td>${r.Diastolic}</td><td>${r.HR}</td><td>${r.Wake_Sleep===0?"Asleep":"Awake"}</td><td><button class="btn btn-danger" style="padding:2px 6px;font-size:11px" onclick="removeReading(${i})">x</button></td></tr>`
  ).join("");
  countEl.innerHTML = `<strong>${manualReadings.length}</strong> readings entered. ${manualReadings.length<6?"Need at least 6.":"Ready!"}`;
  btn.disabled = manualReadings.length < 6;
}

function removeReading(i) { manualReadings.splice(i,1); refreshReadingsTable(); }

async function analyseManualReadings() {
  const btn = document.getElementById("me-analyse-btn");
  btn.disabled = true; btn.textContent = "Analysing...";
  try {
    const apiBase = await window.api.getApiBase();
    const resp = await fetch(`${apiBase}/api/analyse`, {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({
        readings: manualReadings,
        patient_id: document.getElementById("me-patient-id").value||"MANUAL",
        sleep_start: document.getElementById("me-sleep-start").value||"22:00",
        sleep_end: document.getElementById("me-sleep-end").value||"07:00",
      }),
    });
    if (!resp.ok) throw new Error(await resp.text());
    const profile = await resp.json();
    switchView("dashboard");
    const sub = document.querySelector("#view-dashboard .view-subtitle");
    if (sub) sub.textContent = `Manual entry — ${manualReadings.length} readings`;
    renderDashboard(document.getElementById("dashboard-container"), profile);
  } catch(e) { alert("Analysis failed: "+e.message); }
  finally { btn.disabled=false; btn.textContent="Analyse Readings"; }
}

function exampleReadings() {
  const h=[0,1,2,3,4,5,6,7,8,9,10,12,14,16,18,20,22,23];
  const s=[139,136,138,140,137,139,136,160,164,151,136,133,140,135,142,134,139,137];
  const d=[84,82,83,85,82,84,83,96,99,91,84,80,85,82,86,81,84,83];
  const r=[72,70,69,70,71,70,72,82,86,80,75,73,74,72,76,73,72,71];
  const w=[0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0];
  return h.map((hr,i)=>({Time:`${String(hr).padStart(2,"0")}:00:00`,Systolic:s[i],Diastolic:d[i],HR:r[i],Wake_Sleep:w[i]}));
}
