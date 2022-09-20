FROM ghcr.io/br3ndonland/inboard:fastapi-0.28.0-python3.10

# Copy poetry.lock* in case it doesn't exist in the repo
COPY pyproject.toml poetry.lock prestart.sh /app/

WORKDIR /app

# Allow installing dev dependencies to run tests
ARG INSTALL_DEV=false
# needed for new versions of inboard, see https://github.com/br3ndonland/inboard/discussions/52
ENV POETRY_VIRTUALENVS_IN_PROJECT=true

RUN bash -c "if [ $INSTALL_DEV == 'true' ] ; then poetry install --no-root ; else poetry install --no-root --no-dev ; fi"

ARG BACKEND_APP_MODULE=myapp.main:app
ARG BACKEND_PRE_START_PATH=/app/myapp/prestart.sh
ARG BACKEND_PROCESS_MANAGER=gunicorn
ARG BACKEND_WITH_RELOAD=false

# Inboard envvars are set in the base image
ENV APP_MODULE=${BACKEND_APP_MODULE} PRE_START_PATH=${BACKEND_PRE_START_PATH} PROCESS_MANAGER=${BACKEND_PROCESS_MANAGER} WITH_RELOAD=${BACKEND_WITH_RELOAD}

# copy the rest of the app
COPY myapp /app/myapp
