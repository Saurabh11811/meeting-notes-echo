from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from echo_api.services.meeting_service import create_mom_export, get_meeting_detail, list_meetings, regenerate_meeting_mom

router = APIRouter(prefix="/meetings", tags=["meetings"])


class RegenerateRequest(BaseModel):
    template_name: str = "Executive MoM"
    backend_kind: str = ""
    run_now: bool = True


@router.get("")
def read_meetings() -> dict:
    return {"meetings": list_meetings()}


@router.get("/{meeting_id}")
def read_meeting(meeting_id: str) -> dict:
    meeting = get_meeting_detail(meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")
    return meeting


@router.post("/{meeting_id}/regenerate")
def regenerate_meeting(meeting_id: str, payload: RegenerateRequest) -> dict:
    try:
        return regenerate_meeting_mom(
            meeting_id,
            template_name=payload.template_name,
            backend_kind=payload.backend_kind,
            run_now=payload.run_now,
        )
    except ValueError as exc:
        message = str(exc)
        status = 404 if "not found" in message.lower() else 400
        raise HTTPException(status_code=status, detail=message) from exc


@router.get("/{meeting_id}/exports/{export_type}")
def export_meeting(meeting_id: str, export_type: str) -> FileResponse:
    try:
        path = create_mom_export(meeting_id, export_type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    media_type = {
        "pdf": "application/pdf",
        "email": "message/rfc822",
        "text": "text/plain",
    }.get(export_type, "application/octet-stream")
    return FileResponse(path, media_type=media_type, filename=path.name)
