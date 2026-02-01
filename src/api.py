from fastapi import APIRouter

from src.anomaly.views import router

api_router = APIRouter()

api_router.include_router(router)
