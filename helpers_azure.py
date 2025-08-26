# helpers_azure.py
from __future__ import annotations
import requests

def summarize_azure_openai(
    text: str, *,
    endpoint: str, api_key: str, api_version: str, deployment: str,
    system_prompt: str = "",
) -> str:
    """
    Azure OpenAI chat completions with your deployment name.
    """
    if not text or not text.strip():
        return "No transcript content to summarize."
    if not (endpoint and api_key and api_version and deployment):
        return "[Azure config incomplete]"
    url = f"{endpoint.rstrip('/')}/openai/deployments/{deployment}/chat/completions"
    headers = {"api-key": api_key, "Content-Type": "application/json"}

    sys_msg = (system_prompt.strip() or
               "Executive-ready summary. Provide Key Decisions; Action Items (owner & due date); Risks/Blockers; Notes.")

    payload = {
        "messages": [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": text}
        ],
        "temperature": 0.1,
    }
    try:
        r = requests.post(url, headers=headers, params={"api-version": api_version}, json=payload, timeout=120)
        r.raise_for_status()
        j = r.json()
        return j["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[Azure OpenAI error] {e}"

def test_azure_connection(*, endpoint: str, api_key: str, api_version: str, deployment: str) -> tuple[bool, str]:
    url = f"{endpoint.rstrip('/')}/openai/deployments/{deployment}/chat/completions"
    headers = {"api-key": api_key, "Content-Type": "application/json"}
    payload = {
        "messages": [
            {"role": "system", "content": "You are a health check probe."},
            {"role": "user", "content": "ping"}
        ],
        "temperature": 0.0,
    }
    try:
        r = requests.post(url, headers=headers, params={"api-version": api_version}, json=payload, timeout=30)
        r.raise_for_status()
        _ = r.json()["choices"][0]["message"]["content"]
        return True, "OK"
    except Exception as e:
        return False, f"{e}"
