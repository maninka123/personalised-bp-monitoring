/**
 * build-python.js — Bundle the Python backend with PyInstaller.
 * Run: node build-python.js
 */
const { execSync } = require("child_process");
const path = require("path");
const fs = require("fs");

const BACKEND_DIR = path.join(__dirname, "python-backend");
const repoCandidates = [
  path.join(__dirname, ".."),
  path.join(__dirname, "..", "personalised-bp-monitoring"),
];
const PERSONALISED = repoCandidates.find(candidate =>
  fs.existsSync(path.join(candidate, "clinical_report_utils.py"))
);
if (!PERSONALISED) {
  throw new Error("Could not find personalised-bp-monitoring analysis modules.");
}

console.log("=== Building Python backend with PyInstaller ===");

// Check dependencies without forcing a reinstall. Reinstalling in a large
// Anaconda environment can fail on unrelated broken package metadata.
console.log("[1/3] Checking Python dependencies...");
try {
  execSync(
    `python -c "import fastapi, uvicorn, pydantic, pandas, numpy, matplotlib, openpyxl; import PyInstaller"`,
    { cwd: BACKEND_DIR, stdio: "inherit" }
  );
} catch {
  console.log("Missing dependency detected. Installing backend requirements...");
  execSync(`pip install -r requirements.txt pyinstaller`, {
    cwd: BACKEND_DIR,
    stdio: "inherit",
  });
}

// Build with PyInstaller
console.log("[2/3] Running PyInstaller in onedir mode...");
fs.rmSync(path.join(BACKEND_DIR, "dist"), { recursive: true, force: true });
fs.rmSync(path.join(BACKEND_DIR, "build"), { recursive: true, force: true });
const addData = [
  `${path.join(PERSONALISED, "clinical_report_utils.py")}${path.delimiter}.`,
  `${path.join(PERSONALISED, "sleep_aware_bp_framework.py")}${path.delimiter}.`,
  `${path.join(PERSONALISED, "bp_report_assistant.py")}${path.delimiter}.`,
];
const thresholdsPath = path.join(PERSONALISED, "outputs", "dryad_thresholds.csv");
if (fs.existsSync(thresholdsPath)) {
  addData.push(`${thresholdsPath}${path.delimiter}outputs`);
}
const addDataArgs = addData.map(d => `--add-data "${d}"`).join(" ");
const hiddenImports = [
  "clinical_report_utils",
  "sleep_aware_bp_framework",
  "bp_report_assistant",
  "matplotlib.backends.backend_pdf",
];
const hiddenImportArgs = hiddenImports.map(name => `--hidden-import ${name}`).join(" ");
const projectPathArg = `--paths "${PERSONALISED}"`;

execSync(
  `pyinstaller --onedir --noconfirm --clean --name server --additional-hooks-dir hooks ${projectPathArg} ${hiddenImportArgs} ${addDataArgs} server.py`,
  { cwd: BACKEND_DIR, stdio: "inherit" }
);

console.log("[3/3] Done! Bundled server at python-backend/dist/server/server.exe");
