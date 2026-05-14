from __future__ import annotations

from fastapi import APIRouter, HTTPException

from echo_api.services.meeting_service import get_meeting_detail, list_meetings

router = APIRouter(prefix="/meetings", tags=["meetings"])


@router.get("")
def read_meetings() -> dict:
    return {"meetings": list_meetings()}


@router.get("/{meeting_id}")
def read_meeting(meeting_id: str) -> dict:
    meeting = get_meeting_detail(meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")
    return meeting

