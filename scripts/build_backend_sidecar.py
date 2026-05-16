from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "backend-dist"
BUILD = ROOT / "build" / "pyinstaller"
ENTRYPOINT = ROOT / "backend" / "desktop_server.py"
DEFAULT_CONFIG = ROOT / "backend" / "config" / "default.yaml"
BASE_EXCLUDES = [
    "black",
    "docutils",
    "IPython",
    "jedi",
    "jupyter",
    "matplotlib",
    "nbformat",
    "numpy",
    "pandas",
    "PIL",
    "PyQt5",
    "PyQt6",
    "PySide2",
    "PySide6",
    "pytest",
    "scipy",
    "sphinx",
    "tkinter",
]
ASR_EXCLUDES = [
    "torch",
    "torchaudio",
    "torchvision",
    "transformers",
    "faster_whisper",
]


def main() -> int:
    if not shutil.which("pyinstaller"):
        print("PyInstaller is not installed.")
        print("Install packaging dependencies with:")
        print("  python -m pip install -r requirements-packaging.txt")
        return 1

    DIST.mkdir(parents=True, exist_ok=True)
    separator = ";" if os.name == "nt" else ":"
    add_data = f"{DEFAULT_CONFIG}{separator}backend/config"
    cmd = [
        "pyinstaller",
        "--clean",
        "--noconfirm",
        "--onefile",
        "--name",
        "echo-api",
        "--paths",
        str(ROOT / "backend"),
        "--distpath",
        str(DIST),
        "--workpath",
        str(BUILD / "work"),
        "--specpath",
        str(BUILD / "spec"),
        "--add-data",
        add_data,
    ]
    excludes = list(BASE_EXCLUDES)
    if os.getenv("ECHO_PACKAGE_ASR", "0") not in ("1", "true", "True", "yes", "Yes"):
        excludes.extend(ASR_EXCLUDES)
        print("ASR libraries are excluded. Set ECHO_PACKAGE_ASR=1 to build a larger sidecar with ASR dependencies.")
    else:
        print("ASR packaging is enabled. The sidecar will be larger and may require platform-specific validation.")

    for module in excludes:
        cmd.extend(["--exclude-module", module])
    cmd.append(str(ENTRYPOINT))
    print("Building backend sidecar:")
    print(" ".join(cmd))
    return subprocess.call(cmd, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
