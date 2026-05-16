const { app, BrowserWindow, ipcMain, shell } = require("electron");
const { spawn } = require("node:child_process");
const fs = require("node:fs");
const http = require("node:http");
const path = require("node:path");

const isDev = !app.isPackaged;
const backendHost = process.env.ECHO_BACKEND_HOST || "127.0.0.1";
const backendPort = process.env.ECHO_BACKEND_PORT || "8765";
const backendHealthUrl = `http://${backendHost}:${backendPort}/api/health`;
let backendProcess = null;

function logDesktop(message) {
  const line = `[${new Date().toISOString()}] ${message}\n`;
  try {
    fs.mkdirSync(app.getPath("userData"), { recursive: true });
    fs.appendFileSync(path.join(app.getPath("userData"), "desktop.log"), line);
  } catch {
    // Keep desktop startup resilient even if logging is unavailable.
  }
  console.log(message);
}

function requestHealth(timeoutMs = 1000) {
  return new Promise((resolve) => {
    const req = http.get(backendHealthUrl, { timeout: timeoutMs }, (res) => {
      res.resume();
      resolve(res.statusCode >= 200 && res.statusCode < 500);
    });
    req.on("timeout", () => {
      req.destroy();
      resolve(false);
    });
    req.on("error", () => resolve(false));
  });
}

async function waitForBackend(attempts = 40) {
  for (let index = 0; index < attempts; index += 1) {
    if (await requestHealth(1000)) return true;
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  return false;
}

function platformPathEntries() {
  const entries = [];
  const bundledBin = path.join(process.resourcesPath || "", "bin");
  if (fs.existsSync(bundledBin)) entries.push(bundledBin);

  if (process.platform === "darwin") {
    entries.push("/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/bin");
  } else if (process.platform === "win32") {
    const programFiles = process.env.ProgramFiles || "C:\\Program Files";
    const programFilesX86 = process.env["ProgramFiles(x86)"] || "C:\\Program Files (x86)";
    const localAppData = process.env.LocalAppData || "";
    entries.push(
      path.join(programFiles, "Ollama"),
      path.join(localAppData, "Programs", "Ollama"),
      path.join(programFiles, "ffmpeg", "bin"),
      path.join(programFilesX86, "ffmpeg", "bin"),
      "C:\\ffmpeg\\bin",
    );
  } else {
    entries.push("/usr/local/bin", "/usr/bin", "/bin", "/snap/bin");
  }
  return entries.filter(Boolean);
}

function mergedPath() {
  const current = process.env.PATH || "";
  const existing = new Set(current.split(path.delimiter).filter(Boolean));
  const additions = platformPathEntries().filter((entry) => !existing.has(entry));
  return [...additions, current].filter(Boolean).join(path.delimiter);
}

function backendCommand() {
  if (isDev) {
    const python = process.env.PYTHON_BIN || (process.platform === "win32" ? "python" : "python3");
    return {
      command: python,
      args: ["-m", "echo_api"],
      cwd: path.join(app.getAppPath(), "backend"),
      env: {
        PYTHONPATH: path.join(app.getAppPath(), "backend"),
      },
    };
  }

  const binary = process.platform === "win32" ? "echo-api.exe" : "echo-api";
  return {
    command: path.join(process.resourcesPath, "backend", binary),
    args: [],
    cwd: process.resourcesPath,
    env: {},
  };
}

async function ensureBackend() {
  if (process.env.ECHO_SKIP_BACKEND_SIDECAR === "1") {
    logDesktop("Backend sidecar skipped by ECHO_SKIP_BACKEND_SIDECAR.");
    return;
  }

  if (await requestHealth()) {
    logDesktop(`Backend already running at ${backendHealthUrl}.`);
    return;
  }

  const spec = backendCommand();
  if (!isDev && !fs.existsSync(spec.command)) {
    logDesktop(`Packaged backend sidecar not found at ${spec.command}.`);
    return;
  }

  const dataDir = path.join(app.getPath("userData"), "data");
  fs.mkdirSync(dataDir, { recursive: true });

  const env = {
    ...process.env,
    ...spec.env,
    PATH: mergedPath(),
    ECHO_BUNDLED_BIN_DIR: path.join(process.resourcesPath || "", "bin"),
    ECHO_BACKEND_HOST: backendHost,
    ECHO_BACKEND_PORT: backendPort,
    ECHO_APP_DATA_DIR: dataDir,
  };

  logDesktop(`Starting backend sidecar: ${spec.command} ${spec.args.join(" ")}`);
  backendProcess = spawn(spec.command, spec.args, {
    cwd: spec.cwd,
    env,
    stdio: ["ignore", "pipe", "pipe"],
    windowsHide: true,
  });

  backendProcess.stdout.on("data", (chunk) => logDesktop(`[backend] ${chunk.toString().trimEnd()}`));
  backendProcess.stderr.on("data", (chunk) => logDesktop(`[backend] ${chunk.toString().trimEnd()}`));
  backendProcess.on("exit", (code, signal) => {
    logDesktop(`Backend sidecar exited with code=${code} signal=${signal || ""}.`);
    backendProcess = null;
  });

  const ready = await waitForBackend();
  logDesktop(ready ? `Backend ready at ${backendHealthUrl}.` : `Backend did not become ready at ${backendHealthUrl}.`);
}

function createWindow() {
  const win = new BrowserWindow({
    width: 1440,
    height: 960,
    minWidth: 1100,
    minHeight: 720,
    title: "ECHO",
    backgroundColor: "#0f172a",
    show: false,
    webPreferences: {
      preload: path.join(__dirname, "preload.cjs"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
    },
  });

  win.once("ready-to-show", () => {
    win.show();
  });

  win.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: "deny" };
  });

  const devUrl = process.env.ECHO_UI_DEV_URL || "http://127.0.0.1:5173";
  if (isDev) {
    win.loadURL(devUrl);
    if (process.env.ECHO_ELECTRON_DEVTOOLS === "1") {
      win.webContents.openDevTools({ mode: "detach" });
    }
    return;
  }

  win.loadFile(path.join(app.getAppPath(), "ui", "dist", "index.html"));
}

app.whenReady().then(async () => {
  ipcMain.handle("echo:openExternal", async (_event, url) => {
    if (typeof url !== "string" || !/^https?:\/\//i.test(url)) {
      return { ok: false, message: "Only HTTP and HTTPS URLs can be opened." };
    }
    await shell.openExternal(url);
    return { ok: true };
  });

  await ensureBackend();
  createWindow();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on("window-all-closed", () => {
  if (backendProcess) {
    backendProcess.kill();
    backendProcess = null;
  }
  if (process.platform !== "darwin") {
    app.quit();
  }
});

app.on("before-quit", () => {
  if (backendProcess) {
    backendProcess.kill();
    backendProcess = null;
  }
});
