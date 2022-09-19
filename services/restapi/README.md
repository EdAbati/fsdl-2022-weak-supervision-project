# FSDL 2022 Weak supervision demo model

## Getting started

1. Clone the repository
2. Copy `.env.sample` to `.env` and fill in the values marked with `<CHANGEME>`
3. Run `make dev.all.up` to start the application
   1. If `make` is not available, run `docker-compose -f docker-compose.yml -f docker-compose.override.yml up -d`
4. Go to `http://localhost/docs` to see the application documentation running

## Troubleshooting

### Container exits with: `'inboard.start' is not available`

If the `restapi` container is exiting with a message like

```plain
Error while finding module specification for 'inboard.start' (ModuleNotFoundError: No module named 'inboard')
```

Then you need to access to the container using a temporal shell. To do so, run `make dev.restapi.tempshell` or altenatively

```bash
docker-compose -f docker-compose.yml -f docker-compose.override.yml run --rm --entrypoint bash restapi
```

Then, inside the container, run

```bash
poetry update && poetry install
```

and restart the container with

```bash
docker-compose -f docker-compose.yml -f docker-compose.override.yml up -d
```
