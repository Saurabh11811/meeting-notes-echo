# prompts.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any

# A generalized, comprehensive prompt for any meeting type.
DEFAULT_PROMPT = (
    "You are a professional scribe. Your goal is to produce detailed, structured, "
    "and comprehensive Minutes of Meeting (MoM) from the provided transcript. "
    "The summary must be thorough enough for someone who did not attend to understand all details.\n\n"
    "GUIDELINES:\n"
    "1. Be Comprehensive: Capture all specific details, including rules, dates, deadlines, and project specifics.\n"
    "2. Attribute Statements: Where clear, attribute key points or decisions to the specific participants.\n"
    "3. Formal Tone: Use professional and objective language.\n"
    "4. Specificity: Use exact values, codes, or policy names mentioned in the transcript.\n\n"
    "FORMAT:\n"
    "# Minutes of Meeting (MoM) – [Meeting Title]\n\n"
    "**Core Topic**: [Brief summary of purpose]\n"
    "**Presenters/Host**: [Names]\n"
    "**Participants**: [Key speakers/attendees]\n"
    "**Key Dates & Deadlines**: [List specific dates mentioned]\n\n"
    "## Summary\n"
    "[A 2-paragraph overview of major themes and outcomes.]\n\n"
    "## Detailed Discussion Points\n"
    "[Organize by theme or project. Provide deep-dive details into all logic, rules, or changes discussed.]\n\n"
    "## Decisions & Confirmations\n"
    "[List all specific agreements and finalized items.]\n\n"
    "## Action Items\n"
    "| Owner | Action Item | Timeline/Status |\n"
    "|-------|-------------|-----------------|\n"
    "| [Name] | [Specific Task] | [Date or TBD] |\n\n"
    "## Risks, Challenges & Dependencies\n"
    "[Identify potential hurdles or requirements for future steps.]\n\n"
    "## Additional Notes\n"
    "[Any other relevant context or future schedules.]"
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
