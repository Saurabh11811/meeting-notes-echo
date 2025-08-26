#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 23:39:52 2025

@author: saurabh.agarwal
"""

# chat_helper.py
from __future__ import annotations
from pathlib import Path
import re, time, json, os
from typing import Tuple, List
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

# Defaults are adjustable via function parameters
DEFAULT_PROFILE = str(Path.home() / ".pw-teams-profile")

def start_aggressive_pause(page, enforce_pause_ms: int):
    if enforce_pause_ms <= 0:
        return
    js = f"""
(() => {{
  const PAUSE_EVERY_MS = {enforce_pause_ms};
  const pauseAll = (root) => {{
    const vids = root.querySelectorAll ? root.querySelectorAll('video') : [];
    vids.forEach(v => {{
      try {{
        v.pause();
        v.autoplay = false;
        v.muted = true;
        v.playbackRate = 1.0;
        v.onplay = () => {{ try{{ v.pause(); }}catch(e){{}} }};
      }} catch(e) {{}}
    }});
  }};
  const tick = () => {{
    try {{
      pauseAll(document);
      document.querySelectorAll('iframe').forEach(fr => {{
        try {{
          if (fr.contentDocument) pauseAll(fr.contentDocument);
        }} catch(e) {{}}
      }});
    }} catch(e) {{}}
  }};
  window.__pauseInterval && clearInterval(window.__pauseInterval);
  tick();
  window.__pauseInterval = setInterval(tick, PAUSE_EVERY_MS);
}})();
"""
    try: page.evaluate(js)
    except Exception: pass

def stop_aggressive_pause(page):
    try: page.evaluate("() => { if (window.__pauseInterval) clearInterval(window.__pauseInterval); }")
    except Exception: pass

def find_transcript_container(page):
    """
    Traverse DOM + Shadow DOM to find the largest scrollable region that looks like the transcript list.
    Returns ElementHandle or None. Does NOT click anything.
    """
    js = r"""
(() => {
  const isScroll = el => {
    const s = getComputedStyle(el);
    return (s.overflowY === 'auto' || s.overflowY === 'scroll') && el.scrollHeight > el.clientHeight + 10;
  };
  const looksLikeList = el => el.getAttribute?.('role') === 'list';
  const looksTranscripty = el => {
    const a = (el.getAttribute?.('aria-label')||'') + ' ' + (el.getAttribute?.('title')||'') + ' ' + (el.className||'');
    const t = el.textContent || '';
    const hay = (a + ' ' + t).toLowerCase();
    return hay.includes('transcript') || hay.includes('captions') || hay.includes('subtitles');
  };
  const seen = new Set();
  const stack = [document];
  let best = null, bestArea = 0;
  while (stack.length) {
    const root = stack.pop();
    const all = root.querySelectorAll ? root.querySelectorAll('*') : [];
    for (const el of all) {
      if (seen.has(el)) continue;
      seen.add(el);
      if (el.shadowRoot) stack.push(el.shadowRoot);
      if (isScroll(el) && (looksTranscripty(el) || looksLikeList(el))) {
        const area = el.clientWidth * el.clientHeight;
        if (area > bestArea) { bestArea = area; best = el; }
      }
    }
  }
  return best;
})()
"""
    h = page.evaluate_handle(js)
    return h.as_element()

def collect_visible_text(container_el):
    txt = container_el.evaluate(r"""
(el) => {
  const lines = [];
  const tsRe = /^\d{1,2}:\d{2}(:\d{2}(\.\d{3})?)?$/;
  const walk = (node) => {
    if (!node) return;
    if (node.nodeType === Node.TEXT_NODE) {
      const s = (node.nodeValue || '').trim();
      if (s && !tsRe.test(s)) lines.push(s);
    }
    if (node.shadowRoot) node.shadowRoot.childNodes.forEach(walk);
    node.childNodes && node.childNodes.forEach(walk);
  };
  walk(el);
  const out = [];
  for (const s of lines) { if (!out.length || s != out[out.length-1]) out.push(s); }
  return out;
}
""")
    return txt

def robust_scroll_collect(container_el, page, max_scrolls: int, scroll_pause_ms: int, stabilize_rounds: int):
    # make scrolling deterministic
    try: container_el.evaluate("(el)=>{ el.style.scrollBehavior='auto'; }")
    except Exception: pass

    all_lines, seen = [], set()
    last_top, stable_rounds = -1, 0
    last_good_top = 0

    for _ in range(max_scrolls):
        # 1) capture current view
        chunk = collect_visible_text(container_el)
        for line in chunk:
            if line not in seen:
                seen.add(line); all_lines.append(line)

        # 2) scroll down
        try:
            container_el.evaluate("(el)=>{ el.scrollTop = el.scrollTop + Math.floor(el.clientHeight*0.9); }")
        except Exception:
            pass
        page.wait_for_timeout(scroll_pause_ms)

        # 3) read positions
        try:
            top   = container_el.evaluate("(el)=>el.scrollTop")
            height= container_el.evaluate("(el)=>el.scrollHeight")
            client= container_el.evaluate("(el)=>el.clientHeight")
        except Exception:
            top, height, client = 0, 0, 0

        # anti-reset
        if last_top >= 0 and top + 25 < last_top:
            try:
                container_el.evaluate("(el, t)=>{ el.scrollTop = t; }", last_good_top)
            except Exception:
                pass
            page.wait_for_timeout(int(scroll_pause_ms * 1.2))
            try:
                top = container_el.evaluate("(el)=>el.scrollTop")
            except Exception:
                pass

        if top <= last_top + 2:
            stable_rounds += 1
        else:
            stable_rounds = 0
            last_good_top = top
        last_top = top

        at_bottom = height and (height - top - client) < 2
        if at_bottom and stable_rounds >= stabilize_rounds:
            break

    return all_lines

def sniff_transcript_payloads(page, seconds=6) -> List[str]:
    lines = []
    def on_response(resp):
        try:
            url = resp.url.lower()
            ct  = (resp.headers.get("content-type") or "").lower()
            if "substrate.office.com" in url and ("init?" in url or "warmup" in url or "recommendations" in url):
                return
            if url.endswith(".vtt") or "text/vtt" in ct or ("text/plain" in ct and url.endswith(".vtt")):
                b = resp.body()
                for raw in b.decode("utf-8", errors="ignore").splitlines():
                    s = raw.strip()
                    if not s or s.startswith(("WEBVTT","NOTE","STYLE","Region:","Kind:")): continue
                    if re.match(r"^\d{2}:\d{2}:\d{2}\.\d{3}\s*-->", s): continue
                    lines.append(s)
            elif url.endswith(".srt") or "application/x-subrip" in ct or "subrip" in ct:
                b = resp.body()
                for raw in b.decode("utf-8", errors="ignore").splitlines():
                    s = raw.strip()
                    if not s or s.isdigit() or re.match(r"^\d{2}:\d{2}:\d{2},\d{3}\s*-->", s): continue
                    lines.append(s)
            elif ("transcript" in url or "caption" in url or "subtitles" in url) and "json" in ct:
                b = resp.body()
                try:
                    data = json.loads(b.decode("utf-8", errors="ignore"))
                except Exception:
                    return
                if isinstance(data, dict):
                    if "CombinedRecognizedPhrases" in data:
                        for p in data["CombinedRecognizedPhrases"]:
                            t = p.get("Display") or p.get("Text")
                            if t: lines.append(t)
                    for key in ("segments","lines","items","captions"):
                        if key in data and isinstance(data[key], list):
                            for s in data[key]:
                                if isinstance(s, dict):
                                    t = s.get("text") or s.get("caption") or s.get("display") or s.get("content")
                                    if t: lines.append(t)
                elif isinstance(data, list):
                    for s in data:
                        if isinstance(s, dict):
                            t = s.get("text") or s.get("caption") or s.get("display")
                            if t: lines.append(t)
                        elif isinstance(s, str):
                            lines.append(s)
        except Exception:
            pass

    page.on("response", on_response)
    t0 = time.time()
    while (time.time() - t0) < seconds:
        page.wait_for_timeout(150)

    out, seen = [], set()
    for s in lines:
        if s not in seen:
            seen.add(s); out.append(s)
    return out

def capture_transcript(
    url: str,
    out_dir: str | Path = "./out_transcript",
    sniff_seconds: int = 6,
    enforce_pause_ms: int = 800,
    max_scrolls: int = 320,
    scroll_pause_ms: int = 900,
    stabilize_rounds: int = 3,
    profile_dir: str | None = None,
    close_browser_after_capture: bool = True,
) -> Tuple[str, List[str], List[str]]:
    """
    Opens Teams/Stream viewer URL (no UI clicks), passively sniffs captions and
    scroll-collects the transcript list. Returns (full_text, sniff_lines, dom_lines).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prof = os.path.expanduser(profile_dir or DEFAULT_PROFILE)
    Path(prof).mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(
            user_data_dir=prof,
            headless=False,
            channel="chrome",  # use system Chrome (AAD/SSO stack)
            accept_downloads=True,
            args=["--autoplay-policy=no-user-gesture-required"],
        )
        page = ctx.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=180000)

        start_aggressive_pause(page, enforce_pause_ms)

        sniff_lines = sniff_transcript_payloads(page, sniff_seconds)

        container_el = find_transcript_container(page)
        if not container_el:
            stop_aggressive_pause(page)
            if close_browser_after_capture:
                ctx.close()
            # Return empty to let caller decide UI message
            return ("", sniff_lines, [])

        dom_lines = robust_scroll_collect(container_el, page, max_scrolls, scroll_pause_ms, stabilize_rounds)

        stop_aggressive_pause(page)

        # Merge & dedupe
        combined, seen = [], set()
        for s in (sniff_lines + dom_lines):
            s = (s or "").strip()
            if s and s not in seen:
                seen.add(s); combined.append(s)

        # Save (like your original)
        (out_dir / "transcript_full.txt").write_text("\n".join(combined), encoding="utf-8")
        (out_dir / "transcript_sniff.txt").write_text("\n".join(sniff_lines), encoding="utf-8")
        (out_dir / "transcript_dom.txt").write_text("\n".join(dom_lines), encoding="utf-8")

        if close_browser_after_capture:
            ctx.close()

        return ("\n".join(combined), sniff_lines, dom_lines)
