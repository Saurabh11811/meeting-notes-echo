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
    timeout: int = 180,
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

    base = app_base.rstrip('/')
    url = f"{base}/chat-messages" if base.endswith('/v1') else f"{base}/v1/chat-messages"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "inputs": {},
        "query": query,
        "response_mode": "streaming", # Use streaming to keep connection alive
        "user": user_id,
    }
    
    full_answer = []
    try:
        # We use stream=True and iterate over lines to handle SSE (Server-Sent Events)
        r = requests.post(url, headers=headers, json=payload, timeout=timeout, stream=True)
        
        if r.status_code != 200:
            return f"[Dify native error] {r.status_code} {r.reason}: {r.text}"
            
        import json
        for line in r.iter_lines():
            if not line:
                continue
            line_str = line.decode('utf-8')
            if line_str.startswith('data: '):
                try:
                    data = json.loads(line_str[6:])
                    event = data.get("event")
                    if event == "message":
                        full_answer.append(data.get("answer", ""))
                    elif event == "error":
                        return f"[Dify stream error] {data.get('message')}"
                    elif event == "message_end":
                        break
                except Exception:
                    continue
        
        final_text = "".join(full_answer)
        return final_text or "[Dify native: empty answer]"
        
    except Exception as e:
        return f"[Dify native error] {e}"

def test_dify_connection(*, api_key: str, app_base: str = "https://api.dify.ai") -> tuple[bool, str]:
    """
    Minimal connectivity test: sends a tiny prompt and checks for a response.
    """
    resp = summarize_dify_native("ping", api_key=api_key, app_base=app_base)
    ok = not resp.startswith("[Dify native error]")
    return ok, ("OK" if ok else resp)
