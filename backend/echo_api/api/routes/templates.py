from __future__ import annotations

from typing import List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from echo_api.services.template_service import (
    list_templates,
    get_template,
    create_template,
    update_template,
    delete_template,
    duplicate_template,
    get_template_prompt,
    list_template_presets,
    list_template_versions,
    restore_template_version,
)

router = APIRouter(prefix="/templates", tags=["templates"])

class TemplateBase(BaseModel):
    name: str
    meeting_type: str = "General"
    description: str = ""
    sections: List[str] = Field(default_factory=list)
    system_prompt: str = ""
    is_default: bool = False

class TemplateCreate(TemplateBase):
    pass

class TemplateUpdate(BaseModel):
    name: Optional[str] = None
    meeting_type: Optional[str] = None
    description: Optional[str] = None
    sections: Optional[List[str]] = None
    system_prompt: Optional[str] = None
    is_default: Optional[bool] = None

@router.get("")
def read_templates():
    return {"templates": list_templates()}


@router.get("/presets")
def read_template_presets():
    return {"presets": list_template_presets()}

@router.get("/{template_id}")
def read_template(template_id: str):
    template = get_template(template_id)
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")
    return template


@router.get("/{template_id}/prompt")
def read_prompt(template_id: str):
    prompt = get_template_prompt(template_id)
    if not prompt:
        raise HTTPException(status_code=404, detail="Template not found")
    return prompt


@router.get("/{template_id}/effective-prompt")
def read_effective_prompt(template_id: str):
    prompt = get_template_prompt(template_id)
    if not prompt:
        raise HTTPException(status_code=404, detail="Template not found")
    return prompt


@router.get("/{template_id}/versions")
def read_template_versions(template_id: str):
    template = get_template(template_id)
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")
    return {"versions": list_template_versions(template_id)}


@router.post("/{template_id}/versions/{version_id}/restore")
def post_restore_template_version(template_id: str, version_id: str):
    template = restore_template_version(template_id, version_id)
    if not template:
        raise HTTPException(status_code=404, detail="Template version not found")
    return template

@router.post("")
def post_template(payload: TemplateCreate):
    return create_template(payload.model_dump())

@router.patch("/{template_id}")
def patch_template(template_id: str, payload: TemplateUpdate):
    template = update_template(template_id, payload.model_dump(exclude_unset=True))
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")
    return template

@router.post("/{template_id}/duplicate")
def post_duplicate_template(template_id: str):
    template = duplicate_template(template_id)
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")
    return template

@router.delete("/{template_id}")
def remove_template(template_id: str):
    success = delete_template(template_id)
    if not success:
        raise HTTPException(status_code=400, detail="Could not delete template (it may be locked or not found)")
    return {"status": "deleted"}
