# helpers_dify.py
from __future__ import annotations
import requests

def summarize_dify_native(
    text: str,
    *,
    api_key: str,
    app_base: str = "https://api.dify.ai",
    user_id: str = "meeting-notes-mvp",
    system_prompt: str = "",
) -> str:
    """
    Dify native App API (no model param). Provide your App API key & base URL.
    Docs: https://docs.dify.ai/reference/app-api
    """
    if not text or not text.strip():
        return "No transcript content to summarize."
    if not api_key:
        return "[Dify API key missing]"

    # If your Dify app expects a system prompt, many setups read it from the first message of the session.
    # Native API doesn't have a separate 'system' field, so we prepend it to the query if provided.
    query = text if not system_prompt.strip() else f"[SYSTEM]\n{system_prompt.strip()}\n\n[TRANSCRIPT]\n{text}"

    url = f"{app_base.rstrip('/')}/v1/chat-messages"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "inputs": {},
        "query": query,
        "response_mode": "blocking",
        "user": user_id,
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=120)
        r.raise_for_status()
        j = r.json()
        return j.get("answer", "") or "[Dify native: empty answer]"
    except Exception as e:
        return f"[Dify native error] {e}"

def test_dify_connection(*, api_key: str, app_base: str = "https://api.dify.ai") -> tuple[bool, str]:
    """
    Minimal connectivity test: sends a tiny prompt and checks for a response.
    """
    resp = summarize_dify_native("ping", api_key=api_key, app_base=app_base)
    ok = not resp.startswith("[Dify native error]")
    return ok, ("OK" if ok else resp)
