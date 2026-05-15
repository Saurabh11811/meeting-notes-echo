from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from echo_api.services.job_service import create_jobs, create_upload_jobs, list_jobs
from echo_api.services.processing_service import process_available_jobs, process_next_job, retry_failed_jobs, start_job

router = APIRouter(prefix="/jobs", tags=["jobs"])


class JobCreateRequest(BaseModel):
    source_type: Literal["url", "upload", "transcript"]
    sources: list[str] = Field(default_factory=list)
    meeting_type: str = "Executive"
    template_name: str = "Executive MoM"
    confidentiality: str = "Internal"
    project: str = ""
    host: str = ""
    run_now: bool = True


@router.get("")
def read_jobs() -> dict:
    return {"jobs": list_jobs()}


@router.post("/process-next")
def process_next_queue_job() -> dict:
    job = process_next_job()
    return {
        "started": job,
        "message": "Started the next queued job." if job else "No queued job was ready to start.",
    }


@router.post("/process-available")
def process_available_queue_jobs() -> dict:
    jobs = process_available_jobs()
    return {
        "started": jobs,
        "message": f"Started {len(jobs)} queued job{'s' if len(jobs) != 1 else ''} sequentially.",
    }


@router.post("")
def create_queue_jobs(payload: JobCreateRequest) -> dict:
    try:
        return {"jobs": create_jobs(payload.model_dump())}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/retry-failed")
def retry_failed_queue_jobs() -> dict:
    jobs = retry_failed_jobs()
    return {
        "started": jobs,
        "message": f"Retried {len(jobs)} failed job{'s' if len(jobs) != 1 else ''}.",
    }


@router.post("/{job_id}/start")
def start_queue_job(job_id: str) -> dict:
    job = start_job(job_id)
    if not job:
        raise HTTPException(status_code=409, detail="Job is not queued or is already being processed.")
    return {"job": job}


@router.post("/upload")
async def create_upload_queue_jobs(
    files: list[UploadFile] = File(...),
    meeting_type: str = Form("Executive"),
    template_name: str = Form("Executive MoM"),
    confidentiality: str = Form("Internal"),
    project: str = Form(""),
    host: str = Form(""),
    run_now: bool = Form(True),
) -> dict:
    try:
        return {
            "jobs": await create_upload_jobs(
                files,
                {
                    "meeting_type": meeting_type,
                    "template_name": template_name,
                    "confidentiality": confidentiality,
                    "project": project,
                    "host": host,
                    "run_now": run_now,
                },
            )
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
