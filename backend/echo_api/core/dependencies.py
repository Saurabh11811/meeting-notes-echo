from __future__ import annotations

import os
import platform
import shutil
from pathlib import Path
from typing import Iterable


def _split_path(value: str | None) -> list[str]:
    if not value:
        return []
    return [item for item in value.split(os.pathsep) if item]


def bundled_bin_dir() -> Path | None:
    configured = os.getenv("ECHO_BUNDLED_BIN_DIR")
    if configured:
        return Path(configured).expanduser()
    return None


def common_bin_dirs() -> list[Path]:
    system = platform.system().lower()
    dirs: list[Path] = []
    bundled = bundled_bin_dir()
    if bundled:
        dirs.append(bundled)

    if system == "darwin":
        dirs.extend([
            Path("/opt/homebrew/bin"),
            Path("/usr/local/bin"),
            Path("/usr/bin"),
            Path("/bin"),
        ])
    elif system == "windows":
        program_files = Path(os.getenv("ProgramFiles", r"C:\Program Files"))
        program_files_x86 = Path(os.getenv("ProgramFiles(x86)", r"C:\Program Files (x86)"))
        local_app_data = Path(os.getenv("LocalAppData", ""))
        dirs.extend([
            program_files / "Ollama",
            local_app_data / "Programs" / "Ollama",
            program_files / "ffmpeg" / "bin",
            program_files_x86 / "ffmpeg" / "bin",
            Path(r"C:\ffmpeg\bin"),
        ])
    else:
        dirs.extend([
            Path("/usr/local/bin"),
            Path("/usr/bin"),
            Path("/bin"),
            Path("/snap/bin"),
        ])
    return dirs


def executable_names(name: str) -> list[str]:
    if platform.system().lower() == "windows" and not name.lower().endswith(".exe"):
        return [f"{name}.exe", name]
    return [name]


def find_binary(name: str, *, env_var: str | None = None, extra_dirs: Iterable[Path] = ()) -> str | None:
    if env_var and os.getenv(env_var):
        configured = Path(os.environ[env_var]).expanduser()
        if configured.exists():
            return str(configured)

    located = shutil.which(name)
    if located:
        return located

    for directory in [*extra_dirs, *common_bin_dirs()]:
        if not directory:
            continue
        for executable in executable_names(name):
            candidate = directory / executable
            if candidate.exists():
                return str(candidate)
    return None


def browser_candidates() -> list[tuple[str, str]]:
    configured = os.getenv("ECHO_CHROME_PATH") or os.getenv("PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH")
    candidates: list[tuple[str, str]] = []
    if configured:
        candidates.append(("Configured browser", configured))

    system = platform.system().lower()
    if system == "darwin":
        candidates.extend([
            ("Google Chrome", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
            ("Microsoft Edge", "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"),
            ("Chromium", "/Applications/Chromium.app/Contents/MacOS/Chromium"),
        ])
    elif system == "windows":
        program_files = os.getenv("ProgramFiles", r"C:\Program Files")
        program_files_x86 = os.getenv("ProgramFiles(x86)", r"C:\Program Files (x86)")
        local_app_data = os.getenv("LocalAppData", "")
        candidates.extend([
            ("Google Chrome", rf"{program_files}\Google\Chrome\Application\chrome.exe"),
            ("Google Chrome", rf"{program_files_x86}\Google\Chrome\Application\chrome.exe"),
            ("Google Chrome", rf"{local_app_data}\Google\Chrome\Application\chrome.exe"),
            ("Microsoft Edge", rf"{program_files}\Microsoft\Edge\Application\msedge.exe"),
            ("Microsoft Edge", rf"{program_files_x86}\Microsoft\Edge\Application\msedge.exe"),
            ("Chromium", rf"{program_files}\Chromium\Application\chrome.exe"),
        ])
    else:
        for binary, label in (
            ("google-chrome", "Google Chrome"),
            ("google-chrome-stable", "Google Chrome"),
            ("microsoft-edge", "Microsoft Edge"),
            ("msedge", "Microsoft Edge"),
            ("chromium", "Chromium"),
            ("chromium-browser", "Chromium"),
        ):
            located = shutil.which(binary)
            if located:
                candidates.append((label, located))
    return candidates


def find_browser() -> tuple[str, str] | None:
    for label, candidate in browser_candidates():
        path = Path(candidate).expanduser()
        if path.exists():
            return label, str(path)
    return None


def install_guidance() -> dict[str, dict[str, str]]:
    system = platform.system().lower()
    if system == "darwin":
        return {
            "ollama": {"download_url": "https://ollama.com/download/mac", "install": "Download Ollama for macOS, open it once, then pull the model."},
            "ffmpeg": {"download_url": "https://ffmpeg.org/download.html", "install": "Install via Homebrew: brew install ffmpeg"},
            "browser": {"download_url": "https://www.google.com/chrome/", "install": "Install Google Chrome or Microsoft Edge for meeting transcript capture."},
        }
    if system == "windows":
        return {
            "ollama": {"download_url": "https://ollama.com/download/windows", "install": "Download Ollama for Windows, launch it once, then pull the model."},
            "ffmpeg": {"download_url": "https://www.gyan.dev/ffmpeg/builds/", "install": "Install ffmpeg and add its bin folder to PATH."},
            "browser": {"download_url": "https://www.google.com/chrome/", "install": "Install Chrome or Edge. Edge is usually already present on Windows."},
        }
    return {
        "ollama": {"download_url": "https://ollama.com/download/linux", "install": "Install Ollama with the official Linux command, then pull the model."},
        "ffmpeg": {"download_url": "https://ffmpeg.org/download.html", "install": "Install via your distro package manager, for example: sudo apt install ffmpeg"},
        "browser": {"download_url": "https://www.google.com/chrome/", "install": "Install Chrome, Edge, or Chromium for transcript capture."},
    }
