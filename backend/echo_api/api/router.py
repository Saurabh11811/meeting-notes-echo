from fastapi import APIRouter

from echo_api.api.routes import health, home, jobs, meetings, settings

api_router = APIRouter(prefix="/api")
api_router.include_router(health.router)
api_router.include_router(settings.router)
api_router.include_router(home.router)
api_router.include_router(jobs.router)
api_router.include_router(meetings.router)
