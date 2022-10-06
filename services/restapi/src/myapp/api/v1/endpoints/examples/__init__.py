from fastapi import APIRouter

from myapp.api.v1.endpoints.examples import hello

router = APIRouter()

router.include_router(hello.router, prefix="/examples", tags=["examples"])
