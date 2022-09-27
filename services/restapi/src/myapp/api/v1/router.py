from fastapi import APIRouter
from myapp.api.v1.endpoints import examples

__doc__ = "main router for the API"

api_router = APIRouter()
api_router.include_router(examples.router, tags=["demo"])
