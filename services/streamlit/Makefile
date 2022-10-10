.DEFAULT_GOAL := help

# help command will parse and color strings after '##' as documentation automatically
# see https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html for details

.PHONY: help
help: ## Print this help
	@grep -E '^[0-9a-zA-Z_\.-]+:.*?## .*$$' Makefile | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.PHONY: conda-env-and-update
conda-env-and-update: # Create and update a virtual environment using conda
	conda env update --prune -f environment.yml
	@echo "!!! PLEASE ACTIVATE CONDA ENVIRONMENT !!!"

pip-tools: # Compile and install exact pip packages
	pip-compile requirements.in
	pip-sync requirements.txt
