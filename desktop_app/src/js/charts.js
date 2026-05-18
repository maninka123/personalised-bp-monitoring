/* ── charts.js — Chart.js wrapper for BP charts ────────────────── */

const CHARTJS_CDN = "https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js";
let chartJsReady = false;

function loadChartJs() {
  return new Promise((resolve, reject) => {
    if (typeof Chart !== "undefined") { chartJsReady = true; resolve(); return; }
    const s = document.createElement("script");
    s.src = CHARTJS_CDN;
    s.onload = () => { chartJsReady = true; resolve(); };
    s.onerror = () => reject(new Error("Failed to load Chart.js"));
    document.head.appendChild(s);
  });
}

const COLORS = {
  primary: "#2d8cf0",
  sbp: "#1e5f8f",
  dbp: "#7b587b",
  green: "#22c55e",
  yellow: "#eab308",
  red: "#ef4444",
  muted: "#94a3b8",
  grid: "rgba(0,0,0,0.05)",
};

const FONT = "'Inter', 'Segoe UI', sans-serif";

/* ── Shared chart defaults ─────────────────────────────────────── */
function baseOpts() {
  return {
    responsive: true,
    maintainAspectRatio: false,
    resizeDelay: 50,
    plugins: {
      legend: {
        position: "top",
        labels: { usePointStyle: true, padding: 14, font: { size: 12, family: FONT } },
      },
      tooltip: {
        backgroundColor: "#1e293b",
        titleFont: { family: FONT, size: 12 },
        bodyFont: { family: FONT, size: 12 },
        cornerRadius: 6,
        padding: 10,
      },
    },
    animation: { duration: 700, easing: "easeOutQuart" },
  };
}

/* ═══════════════════════════════════════════════════════════════════
   1. 24-hour BP curve
   ═══════════════════════════════════════════════════════════════════ */
function render24hChart(canvas, profile) {
  if (!chartJsReady || !profile.readings) return null;
  const readings = profile.readings;
  const hours = readings.map(r => r.hours_since_start);
  const sbp = readings.map(r => r.Systolic);
  const dbp = readings.map(r => r.Diastolic);

  // Color dots by sleep state
  const sbpBg = readings.map(r => r.Wake_Sleep === 0 ? "rgba(99,102,241,0.85)" : COLORS.sbp);
  const dbpBg = readings.map(r => r.Wake_Sleep === 0 ? "rgba(139,92,139,0.85)" : COLORS.dbp);

  // Sleep background shading plugin
  const sleepBgPlugin = {
    id: "sleepBg",
    beforeDraw(chart) {
      const { ctx, chartArea, scales } = chart;
      if (!chartArea) return;
      ctx.save();
      readings.forEach((r, i) => {
        if (r.Wake_Sleep !== 0) return;
        const xPixel = scales.x.getPixelForValue(i);
        const halfBar = (chartArea.width / readings.length) / 2;
        ctx.fillStyle = "rgba(99,102,241,0.06)";
        ctx.fillRect(xPixel - halfBar, chartArea.top, halfBar * 2, chartArea.bottom - chartArea.top);
      });
      ctx.restore();
    },
  };

  const opts = baseOpts();
  opts.plugins.tooltip.callbacks = {
    title: (items) => {
      const i = items[0].dataIndex;
      const r = readings[i];
      return r.time_display || `${hours[i].toFixed(1)}h`;
    },
    label: ctx => `${ctx.dataset.label}: ${ctx.parsed.y} mmHg`,
    afterBody: (items) => {
      const r = readings[items[0].dataIndex];
      return r.Wake_Sleep === 0 ? "🌙 Asleep" : "☀️ Awake";
    },
  };
  opts.scales = {
    x: {
      grid: { display: false },
      ticks: { font: { size: 10, family: FONT }, maxRotation: 0, autoSkipPadding: 12 },
      title: { display: true, text: "Hours since first reading", font: { size: 12, family: FONT } },
    },
    y: {
      grid: { color: COLORS.grid },
      ticks: { font: { size: 11, family: FONT } },
      title: { display: true, text: "BP (mmHg)", font: { size: 12, family: FONT } },
    },
  };

  return new Chart(canvas, {
    type: "line",
    data: {
      labels: hours.map(h => h.toFixed(1) + "h"),
      datasets: [
        {
          label: "Top number (SBP)",
          data: sbp,
          borderColor: COLORS.sbp,
          backgroundColor: COLORS.sbp,
          pointBackgroundColor: sbpBg,
          pointBorderColor: sbpBg,
          pointRadius: 5,
          pointHoverRadius: 8,
          tension: 0.3,
          borderWidth: 2.5,
        },
        {
          label: "Bottom number (DBP)",
          data: dbp,
          borderColor: COLORS.dbp,
          backgroundColor: COLORS.dbp,
          pointBackgroundColor: dbpBg,
          pointBorderColor: dbpBg,
          pointRadius: 4,
          pointHoverRadius: 7,
          tension: 0.3,
          borderWidth: 2,
        },
      ],
    },
    options: opts,
    plugins: [sleepBgPlugin],
  });
}

/* ═══════════════════════════════════════════════════════════════════
   2. Awake vs Sleep bar chart
   ═══════════════════════════════════════════════════════════════════ */
function renderAwakeSleepChart(canvas, profile) {
  if (!chartJsReady) return null;

  const sbp = [profile.awake_mean_sbp, profile.sleep_mean_sbp];
  const dbp = [profile.awake_mean_dbp, profile.sleep_mean_dbp];

  // Threshold line plugin
  const thresholdPlugin = {
    id: "thresholds",
    afterDraw(chart) {
      const { ctx, chartArea, scales } = chart;
      if (!chartArea) return;
      ctx.save();
      // SBP awake threshold 135, sleep threshold 120
      const thresholds = [
        { val: 135, label: "Awake SBP limit (135)", lineColor: "rgba(185,28,28,0.72)", textColor: "#991b1b" },
        { val: 120, label: "Sleep SBP limit (120)", lineColor: "rgba(67,56,202,0.72)", textColor: "#3730a3" },
      ];
      thresholds.forEach(t => {
        const y = scales.y.getPixelForValue(t.val);
        if (y < chartArea.top || y > chartArea.bottom) return;
        ctx.setLineDash([6, 4]);
        ctx.strokeStyle = t.lineColor;
        ctx.lineWidth = 1.8;
        ctx.beginPath();
        ctx.moveTo(chartArea.left, y);
        ctx.lineTo(chartArea.right, y);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = t.textColor;
        ctx.font = `600 10.5px ${FONT}`;
        ctx.textAlign = "right";
        ctx.fillText(t.label, chartArea.right - 4, y - 4);
      });
      ctx.restore();
    },
  };

  const opts = baseOpts();
  opts.scales = {
    x: { grid: { display: false }, ticks: { font: { size: 12, family: FONT } } },
    y: {
      grid: { color: COLORS.grid },
      beginAtZero: false,
      title: { display: true, text: "BP (mmHg)", font: { size: 12, family: FONT } },
      ticks: { font: { size: 11, family: FONT } },
    },
  };

  return new Chart(canvas, {
    type: "bar",
    data: {
      labels: ["☀️ Awake", "🌙 Sleep"],
      datasets: [
        {
          label: "Top (SBP)",
          data: sbp,
          backgroundColor: [COLORS.sbp, "rgba(99,102,241,0.75)"],
          borderRadius: 6,
          barPercentage: 0.55,
          borderSkipped: false,
        },
        {
          label: "Bottom (DBP)",
          data: dbp,
          backgroundColor: [COLORS.dbp, "rgba(139,92,139,0.65)"],
          borderRadius: 6,
          barPercentage: 0.55,
          borderSkipped: false,
        },
      ],
    },
    options: opts,
    plugins: [thresholdPlugin],
  });
}

/* ═══════════════════════════════════════════════════════════════════
   3. Profile Position — the key chart to match the PDF
   ═══════════════════════════════════════════════════════════════════
   Zones:
     x < 0       → Reverse dipper  (red tint)
     0 ≤ x < 10  → Non-dipper      (yellow/amber tint)
     10 ≤ x < 20 → Normal dipper   (green tint)
     20 ≤ x ≤ 30 → Extreme dipper  (light green tint)
   Threshold:
     y = 20 → High morning surge  (dashed orange line)
   ═══════════════════════════════════════════════════════════════════ */

const DIPPING_ZONES = [
  { xMin: -5, xMax: 0,  color: "rgba(254,202,202,0.35)", label: "Reverse",        labelX: -2.5 },
  { xMin: 0,  xMax: 10, color: "rgba(254,240,138,0.35)", label: "Non-dipper",      labelX: 5    },
  { xMin: 10, xMax: 20, color: "rgba(187,247,208,0.35)", label: "Normal\ndipper",  labelX: 15   },
  { xMin: 20, xMax: 30, color: "rgba(220,252,231,0.30)", label: "Extreme\ndipper", labelX: 25   },
];

const SURGE_THRESHOLD = 20;

function renderProfilePositionChart(canvas, profile) {
  if (!chartJsReady) return null;
  const dip = profile.dipping_pct_sbp;
  const surge = profile.morning_surge_sbp;
  const hasPoint = dip != null && !isNaN(dip) && surge != null && !isNaN(surge);

  // Background zones + threshold plugin
  const zonesPlugin = {
    id: "profileZones",
    beforeDraw(chart) {
      const { ctx, chartArea, scales } = chart;
      if (!chartArea) return;
      ctx.save();

      // Draw colored zones
      DIPPING_ZONES.forEach(zone => {
        const x1 = Math.max(scales.x.getPixelForValue(zone.xMin), chartArea.left);
        const x2 = Math.min(scales.x.getPixelForValue(zone.xMax), chartArea.right);
        ctx.fillStyle = zone.color;
        ctx.fillRect(x1, chartArea.top, x2 - x1, chartArea.bottom - chartArea.top);
      });

      // Zone boundary lines (vertical)
      [0, 10, 20].forEach(xVal => {
        const x = scales.x.getPixelForValue(xVal);
        if (x < chartArea.left || x > chartArea.right) return;
        ctx.setLineDash([]);
        ctx.strokeStyle = "rgba(0,0,0,0.1)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x, chartArea.top);
        ctx.lineTo(x, chartArea.bottom);
        ctx.stroke();
      });

      // Morning surge threshold (horizontal dashed line at y=20)
      const surgeY = scales.y.getPixelForValue(SURGE_THRESHOLD);
      if (surgeY >= chartArea.top && surgeY <= chartArea.bottom) {
        ctx.setLineDash([8, 5]);
        ctx.strokeStyle = "rgba(194,120,3,0.6)";
        ctx.lineWidth = 1.8;
        ctx.beginPath();
        ctx.moveTo(chartArea.left, surgeY);
        ctx.lineTo(chartArea.right, surgeY);
        ctx.stroke();
        ctx.setLineDash([]);

        // Label
        ctx.fillStyle = "rgba(194,120,3,0.7)";
        ctx.font = `11px ${FONT}`;
        ctx.textAlign = "right";
        ctx.fillText("High morning surge region", chartArea.right - 6, surgeY - 6);
      }

      // Zone labels at bottom
      ctx.textAlign = "center";
      ctx.font = `10px ${FONT}`;
      ctx.fillStyle = "rgba(0,0,0,0.35)";
      DIPPING_ZONES.forEach(zone => {
        const x = scales.x.getPixelForValue(zone.labelX);
        const lines = zone.label.split("\n");
        const baseY = chartArea.bottom - 10 - (lines.length - 1) * 13;
        lines.forEach((line, i) => {
          ctx.fillText(line, x, baseY + i * 13);
        });
      });

      ctx.restore();
    },
  };

  // Patient label plugin
  const labelPlugin = {
    id: "patientLabel",
    afterDraw(chart) {
      if (!hasPoint) return;
      const { ctx, scales, chartArea } = chart;
      if (!chartArea) return;
      const px = scales.x.getPixelForValue(dip);
      const py = scales.y.getPixelForValue(surge);
      ctx.save();
      ctx.fillStyle = "#1e293b";
      ctx.font = `bold 11px ${FONT}`;
      ctx.textAlign = "center";
      ctx.fillText("Patient", px, py - 16);
      // connector line
      ctx.strokeStyle = "rgba(30,41,59,0.4)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(px, py - 14);
      ctx.lineTo(px, py - 7);
      ctx.stroke();
      ctx.restore();
    },
  };

  const datasets = [];
  if (hasPoint) {
    datasets.push({
      label: "Patient",
      data: [{ x: dip, y: surge }],
      backgroundColor: "#991b1b",
      borderColor: "#7f1d1d",
      borderWidth: 2,
      pointRadius: 9,
      pointHoverRadius: 12,
      pointStyle: "circle",
    });
  }

  const opts = baseOpts();
  opts.plugins.legend = { display: false };
  opts.plugins.tooltip = {
    ...opts.plugins.tooltip,
    callbacks: {
      title: () => "Patient Profile Position",
      label: ctx => [
        `Sleep BP dipping: ${ctx.parsed.x.toFixed(1)}%`,
        `Morning surge: ${ctx.parsed.y.toFixed(0)} mmHg`,
      ],
    },
  };
  opts.scales = {
    x: {
      type: "linear",
      min: -5,
      max: 30,
      title: { display: true, text: "Sleep SBP dipping (%)", font: { size: 12, family: FONT, weight: "600" } },
      grid: { color: "rgba(0,0,0,0.04)", drawTicks: true },
      ticks: { font: { size: 11, family: FONT }, stepSize: 5 },
    },
    y: {
      type: "linear",
      min: 0,
      max: 35,
      title: { display: true, text: "Morning surge (mmHg)", font: { size: 12, family: FONT, weight: "600" } },
      grid: { color: "rgba(0,0,0,0.04)" },
      ticks: { font: { size: 11, family: FONT }, stepSize: 5 },
    },
  };

  return new Chart(canvas, {
    type: "scatter",
    data: { datasets },
    options: opts,
    plugins: [zonesPlugin, labelPlugin],
  });
}

/* ═══════════════════════════════════════════════════════════════════
   4. Resize handler — makes all charts reflow on window resize
   ═══════════════════════════════════════════════════════════════════ */

let _resizeTimer = null;

function setupChartResize() {
  window.addEventListener("resize", () => {
    if (_resizeTimer) clearTimeout(_resizeTimer);
    _resizeTimer = setTimeout(() => {
      activeCharts.forEach(c => {
        try { c.resize(); } catch {}
      });
    }, 150);
  });
}

// Initialise resize listener once
setupChartResize();
