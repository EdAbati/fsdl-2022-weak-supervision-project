.DEFAULT_GOAL := help

# touch .env if it doesn't exist. .env should be in .gitignore
include .env
export

# help command will parse and color strings after '##' as documentation automatically
# see https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html for details

.PHONY: help
help: ## Print this help
	@grep -E '^[0-9a-zA-Z_\.-]+:.*?## .*$$' Makefile | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.PHONY: conda-env-and-update
conda-env-and-update: ## install all pre-commit hooks
	conda env update --prune -f environment.yml
	@echo "!!! PLEASE ACTIVATE CONDA ENVIRONMENT !!!"

.PHONY: install-pre-commit
install-pre-commit: ## create and update a virtual environment using conda
	pre-commit install --install-hooks

build: ## build docker image
	@docker compose build

push: ## build docker image
	@docker compose push

up: ## start docker-compose
	@docker compose up -d

down: ## stop docker-compose
	@docker compose down

ps: ## get info about running containers
	@docker compose ps

restapi.test: ## run pytest in the restapi
	@docker compose exec restapi pytest

dev.all.up: ## start docker-compose in dev mode
	@docker compose -f docker-compose.yml -f docker-compose.override.yml up -d

dev.all.up.build: ## start docker-compose in dev mode and --build flag to force rebuild
	@docker compose -f docker-compose.yml -f docker-compose.override.yml up -d --build

dev.all.ps: ## start docker-compose ps
	@docker compose -f docker-compose.yml -f docker-compose.override.yml ps

dev.all.down: ## stop docker-compose in dev mode
	@docker compose -f docker-compose.yml -f docker-compose.override.yml down

dev.restapi.shell: ## open shell in restapi container
	@docker compose -f docker-compose.yml -f docker-compose.override.yml exec restapi bash

dev.restapi.test: ## open shell in restapi container
	@docker compose -f docker-compose.yml -f docker-compose.override.yml exec restapi pytest

dev.restapi.tempshell: ## open temp shell in restapi container using run --rm
	@docker compose -f docker-compose.yml -f docker-compose.override.yml run --rm --entrypoint bash restapi

dev.restapi.up: ## launch restapi service in detached mode
	@docker compose -f docker-compose.yml -f docker-compose.override.yml up -d restapi

dev.restapi.down: ## stop restapi service
	@docker compose -f docker-compose.yml -f docker-compose.override.yml down restapi

dev.restapi.build: ## build restapi container
	@docker compose -f docker-compose.yml -f docker-compose.override.yml build restapi

dev.db.shell: ## open shell in restapi container
	@docker compose -f docker-compose.yml -f docker-compose.override.yml exec db bash

dev.restapi.logs: ## fetch logs from restapi container
	@docker compose -f docker-compose.yml -f docker-compose.override.yml logs -ft restapi
