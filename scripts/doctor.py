from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
UI = ROOT / "ui"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from echo_api.core.dependencies import find_binary, find_browser  # noqa: E402


def main() -> int:
    checks: list[dict[str, str | bool]] = []
    checks.extend(check_python())
    checks.extend(check_python_packages())
    checks.extend(check_node())
    checks.extend(check_binaries())
    checks.extend(check_browser())
    checks.extend(check_ollama())
    checks.extend(check_gpu())
    checks.extend(check_storage())

    print("")
    print("ECHO environment doctor")
    print("=" * 72)
    for check in checks:
        mark = "OK" if check["ok"] else ("WARN" if check.get("optional") else "FAIL")
        print(f"[{mark:4}] {check['name']}: {check['detail']}")

    failures = [check for check in checks if not check["ok"] and not check.get("optional")]
    print("=" * 72)
    if failures:
        print(f"{len(failures)} required check(s) failed.")
        return 1
    print("Required checks passed. Review warnings before packaging or demo use.")
    return 0


def check_python() -> list[dict[str, str | bool]]:
    return [
        result(
            "Python",
            sys.version_info >= (3, 10),
            f"{platform.python_version()} at {sys.executable}",
        )
    ]


def check_python_packages() -> list[dict[str, str | bool]]:
    required = [
        ("fastapi", "FastAPI backend"),
        ("uvicorn", "ASGI server"),
        ("yaml", "YAML config"),
        ("pydantic", "API models"),
        ("requests", "HTTP providers"),
        ("playwright.sync_api", "meeting-link browser capture"),
        ("multipart", "file upload parsing"),
        ("ollama", "local AI client"),
        ("truststore", "corporate TLS trust store support"),
    ]
    optional = [
        ("torch", "ASR runtime and GPU detection"),
        ("faster_whisper", "preferred Whisper ASR engine"),
        ("transformers", "HF Whisper fallback"),
    ]
    checks = []
    for module, detail in required:
        checks.append(result(f"Python package {module}", module_available(module), detail))
    for module, detail in optional:
        checks.append(result(f"Python package {module}", module_available(module), detail, optional=True))
    return checks


def check_node() -> list[dict[str, str | bool]]:
    npm = shutil.which("npm")
    node = shutil.which("node")
    checks = [
        result("Node.js", bool(node), run_version([node, "--version"]) if node else "not found"),
        result("npm", bool(npm), run_version([npm, "--version"]) if npm else "not found"),
    ]
    checks.append(result("Frontend dependencies", (UI / "node_modules").exists(), "ui/node_modules present", optional=True))
    return checks


def check_binaries() -> list[dict[str, str | bool]]:
    ffmpeg = find_binary("ffmpeg", env_var="ECHO_FFMPEG_PATH")
    ffprobe = find_binary("ffprobe", env_var="ECHO_FFPROBE_PATH")
    return [
        result("ffmpeg", bool(ffmpeg), ffmpeg or "not found"),
        result("ffprobe", bool(ffprobe), ffprobe or "not found"),
    ]


def check_browser() -> list[dict[str, str | bool]]:
    browser = find_browser()
    detail = f"{browser[0]} at {browser[1]}" if browser else "Chrome/Edge/Chromium not found"
    return [result("Installed browser for transcript capture", bool(browser), detail, optional=True)]


def check_ollama() -> list[dict[str, str | bool]]:
    exe = find_binary("ollama", env_var="ECHO_OLLAMA_PATH")
    checks = [result("Ollama CLI", bool(exe), exe or "not found")]
    if not exe:
        checks.append(result("Ollama model", False, "install Ollama and pull llama3:latest"))
        return checks

    model = configured_ollama_model()
    try:
        proc = subprocess.run([exe, "list"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=8)
        has_model = any(line.strip().startswith(model) for line in proc.stdout.splitlines())
        checks.append(result("Ollama model", has_model, f"{model} {'present' if has_model else 'not found'}"))
    except Exception as exc:
        checks.append(result("Ollama model", False, str(exc)))
    return checks


def check_gpu() -> list[dict[str, str | bool]]:
    if not module_available("torch"):
        return [result("GPU acceleration", False, "torch unavailable; ASR will not run", optional=True)]
    try:
        import torch

        if torch.cuda.is_available():
            return [result("GPU acceleration", True, f"CUDA available: {torch.cuda.get_device_name(0)}", optional=True)]
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return [result("GPU acceleration", True, "Apple MPS available", optional=True)]
        return [result("GPU acceleration", True, "CPU mode", optional=True)]
    except Exception as exc:
        return [result("GPU acceleration", False, str(exc), optional=True)]


def check_storage() -> list[dict[str, str | bool]]:
    target = Path(os.getenv("ECHO_APP_DATA_DIR", BACKEND / ".data")).expanduser()
    try:
        target.mkdir(parents=True, exist_ok=True)
        probe = target / ".doctor_write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return [result("App data storage", True, str(target.resolve()))]
    except Exception as exc:
        return [result("App data storage", False, str(exc))]


def module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def configured_ollama_model() -> str:
    try:
        import yaml

        config_path = Path(os.getenv("ECHO_CONFIG_PATH", BACKEND / ".data" / "config.yaml"))
        if not config_path.exists():
            config_path = BACKEND / "config" / "default.yaml"
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        return str(data.get("backends", {}).get("local", {}).get("model", "llama3:latest"))
    except Exception:
        return "llama3:latest"


def run_version(cmd: list[str]) -> str:
    try:
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=5)
        return proc.stdout.strip() or "installed"
    except Exception as exc:
        return str(exc)


def result(name: str, ok: bool, detail: str, *, optional: bool = False) -> dict[str, str | bool]:
    return {"name": name, "ok": ok, "detail": detail, "optional": optional}


if __name__ == "__main__":
    raise SystemExit(main())
