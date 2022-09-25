from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware

from myapp.api.v1.router import api_router
from myapp.config import settings


def include_routers(app: FastAPI) -> None:
    app.include_router(api_router, prefix=settings.API_V1_STR)


def configure_middleware(app: FastAPI) -> None:
    # Set all CORS enabled origins
    if settings.BACKEND_CORS_ORIGINS:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )


def create_app(
    title: str = settings.PROJECT_NAME,
    openapi_url: str = f"{settings.API_V1_STR}/openapi.json",
    debug: bool = settings.FASTAPI_DEBUG,
):

    app = FastAPI(
        title=title,
        openapi_url=openapi_url,
        debug=debug,
        version=settings.API_DOCKER_TAG,
    )

    include_routers(app)
    configure_middleware(app)

    return app


app = create_app()
