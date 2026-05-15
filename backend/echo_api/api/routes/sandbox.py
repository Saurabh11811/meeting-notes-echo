from __future__ import annotations

from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from echo_api.services.processing_service import summarize_transcript, apply_generation_options
from echo_api.core.config import load_app_config

router = APIRouter(prefix="/sandbox", tags=["sandbox"])

class TestTemplateRequest(BaseModel):
    transcript_text: str
    template_name: str
    backend_kind: Optional[str] = None # e.g. "local", "dify"

@router.post("/test-template")
def test_template(payload: TestTemplateRequest):
    if not payload.transcript_text.strip():
        raise HTTPException(status_code=400, detail="Transcript text is required.")
    
    config = load_app_config()
    
    test_config = apply_generation_options(
        config, 
        backend_kind=payload.backend_kind or config.get("summary", {}).get("default_backend", "local"),
        template_name=payload.template_name
    )
    
    try:
        result = summarize_transcript(payload.transcript_text, test_config)
        return {"output": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
