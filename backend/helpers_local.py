# helpers_local.py
from __future__ import annotations

def summarize_local_ollama(text: str, model: str = "llama3:latest", system_prompt: str = "") -> str:
    """
    Summarize using local Ollama via the official 'ollama' Python library.
    """
    if not text or not text.strip():
        return "No transcript content to summarize."
    
    try:
        import ollama
    except ImportError:
        return "[Local summarizer unavailable: 'ollama' library not installed]"

    sys_prompt = system_prompt.strip() or (
        "Executive-ready summary. Provide Key Decisions, Action Items (owner & due date), Risks/Blockers, Notes."
    )

    try:
        response = ollama.chat(
            model=model,
            messages=[
                {'role': 'system', 'content': sys_prompt},
                {'role': 'user', 'content': text},
            ],
            options={'temperature': 0.1}
        )
        return response['message']['content']
    except Exception as e:
        return f"[Ollama error] {e}"

def test_local_connection(model: str = "llama3:latest") -> tuple[bool, str]:
    """
    Minimal connectivity test for Ollama.
    """
    try:
        import ollama
        # Just check if the model is available or the service is reachable
        _ = ollama.show(model)
        return True, "OK"
    except Exception as e:
        return False, f"{e}"
