#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 21:55:17 2025

@author: saurabh.agarwal
"""

# prompts.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any

DEFAULT_PROMPT = (
    "Executive-ready summary.\n"
    "Provide sections (wherever applicable):\n"
    "1) Key Discussions\n"
    "2) Key Decisions\n"
    "2) Action Items\n"
    "3) Risks / Blockers / Challanges\n"
    "4) Additional Notes\n"
    "Be concise but comprehensive. Use bullet points where helpful."
)

def load_prompt(cfg: Dict[str, Any]) -> str:
    """
    Resolve the system prompt to use:
    1) If summary.prompt_file is a readable path -> file contents
    2) Else if summary.prompt_text is non-empty -> that text
    3) Else -> DEFAULT_PROMPT
    """
    s = cfg.get("summary", {}) if cfg else {}
    path = (s.get("prompt_file") or "").strip()
    if path:
        p = Path(path).expanduser()
        if p.exists() and p.is_file():
            try:
                return p.read_text(encoding="utf-8")
            except Exception:
                pass
    txt = (s.get("prompt_text") or "").strip()
    return txt if txt else DEFAULT_PROMPT
