const { app, BrowserWindow, ipcMain, dialog, shell } = require("electron");
const { spawn } = require("child_process");
const path = require("path");
const net = require("net");
const fs = require("fs");

let mainWindow = null;
let splashWindow = null;
let pythonProcess = null;
const API_PORT = 18347;
const API_BASE = `http://127.0.0.1:${API_PORT}`;

// ── Icon path ─────────────────────────────────────────────────────────────
const ICON_PATH = path.join(__dirname, "build", "icon.ico");

// ── Splash ────────────────────────────────────────────────────────────────
function createSplashWindow() {
  splashWindow = new BrowserWindow({
    width: 520,
    height: 380,
    frame: false,
    transparent: true,
    alwaysOnTop: true,
    resizable: false,
    center: true,
    skipTaskbar: true,
    icon: ICON_PATH,
    webPreferences: { nodeIntegration: false, contextIsolation: true },
  });
  splashWindow.loadFile(path.join(__dirname, "src", "splash.html"));
}

// ── Python backend ────────────────────────────────────────────────────────
function startPythonBackend() {
  const isDev = !app.isPackaged;
  if (isDev) {
    const serverScript = path.join(__dirname, "python-backend", "server.py");
    pythonProcess = spawn("python", [serverScript, "--port", String(API_PORT)], {
      stdio: ["pipe", "pipe", "pipe"],
    });
  } else {
    const serverExe = path.join(process.resourcesPath, "python-backend", "server", "server.exe");
    pythonProcess = spawn(serverExe, ["--port", String(API_PORT)], {
      stdio: ["pipe", "pipe", "pipe"],
    });
  }
  pythonProcess.stdout.on("data", (d) => console.log("[py]", d.toString().trim()));
  pythonProcess.stderr.on("data", (d) => console.error("[py]", d.toString().trim()));
  pythonProcess.on("error", (err) => console.error("[py] spawn error:", err));
  pythonProcess.on("exit", (code) => console.log("[py] exited", code));
}

async function waitForServer(timeoutMs = 30000) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    try {
      await new Promise((resolve, reject) => {
        const sock = new net.Socket();
        sock.setTimeout(400);
        sock.once("connect", () => { sock.destroy(); resolve(); });
        sock.once("error", () => { sock.destroy(); reject(); });
        sock.once("timeout", () => { sock.destroy(); reject(); });
        sock.connect(API_PORT, "127.0.0.1");
      });
      return true;
    } catch {
      await new Promise((r) => setTimeout(r, 350));
    }
  }
  return false;
}

// ── Main window ───────────────────────────────────────────────────────────
function createMainWindow() {
  mainWindow = new BrowserWindow({
    width: 1320,
    height: 860,
    minWidth: 1024,
    minHeight: 700,
    show: false,
    title: "BP Profile Monitor",
    icon: ICON_PATH,
    backgroundColor: "#f8fafc",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      nodeIntegration: false,
      contextIsolation: true,
    },
  });
  mainWindow.loadFile(path.join(__dirname, "src", "index.html"));
  mainWindow.removeMenu();

  mainWindow.once("ready-to-show", () => {
    setTimeout(() => {
      if (splashWindow && !splashWindow.isDestroyed()) splashWindow.close();
      mainWindow.show();
      mainWindow.focus();
    }, 800);
  });
}

// ── IPC handlers ──────────────────────────────────────────────────────────
ipcMain.handle("get-api-base", () => API_BASE);

ipcMain.handle("open-file-dialog", async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    title: "Select ABPM Data File",
    filters: [
      { name: "Data files", extensions: ["csv", "xlsx", "xls"] },
      { name: "All files", extensions: ["*"] },
    ],
    properties: ["openFile"],
  });
  if (result.canceled || !result.filePaths.length) return null;
  const filePath = result.filePaths[0];
  const data = fs.readFileSync(filePath);
  return { name: path.basename(filePath), buffer: data.toString("base64") };
});

ipcMain.handle("open-sample-inputs", async () => {
  const sampleDir = app.isPackaged
    ? path.join(process.resourcesPath, "Sample Patient Inputs")
    : path.join(__dirname, "sample-patient-inputs");
  if (!fs.existsSync(sampleDir)) {
    throw new Error(`Sample input folder not found: ${sampleDir}`);
  }
  await shell.openPath(sampleDir);
  return sampleDir;
});

ipcMain.handle("save-file", async (_event, base64Data, defaultName) => {
  const result = await dialog.showSaveDialog(mainWindow, {
    title: "Save Report",
    defaultPath: defaultName || "bp_report.pdf",
    filters: [{ name: "PDF Report", extensions: ["pdf"] }],
  });
  if (result.canceled || !result.filePath) return null;
  const buffer = Buffer.from(base64Data, "base64");
  fs.writeFileSync(result.filePath, buffer);
  return result.filePath;
});

// ── App lifecycle ─────────────────────────────────────────────────────────
app.whenReady().then(async () => {
  createSplashWindow();
  startPythonBackend();
  const ready = await waitForServer();
  if (!ready) console.warn("Python backend did not start in time — UI will retry.");
  createMainWindow();
});

app.on("window-all-closed", () => {
  if (pythonProcess) { pythonProcess.kill(); pythonProcess = null; }
  app.quit();
});

app.on("before-quit", () => {
  if (pythonProcess) { pythonProcess.kill(); pythonProcess = null; }
});
