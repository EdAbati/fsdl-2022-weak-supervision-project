.DEFAULT_GOAL := help

# touch .env if it doesn't exist. .env should be in .gitignore
include .env
export

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
