from typing import Any

from fastapi import APIRouter


router = APIRouter()


@router.get("/hello", response_model=Any, status_code=201)
def get_about(
    # current_user: models.User = Depends(deps.get_current_active_superuser),
) -> Any:
    """
    Example
    """

    return "Hi there!"
