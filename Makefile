.PHONY: help
help:
	@echo "Commands:"
	@echo "conda-env-and-update    		    : create and update a virtual environment using conda."
	@echo "install-pre-commit       		: install all pre-commit hooks."

.PHONY: conda-env-and-update
conda-env-and-update:
	conda env update --prune -f environment.yml
	@echo "!!! PLEASE ACTIVATE CONDA ENVIRONMENT !!!"

.PHONY: install-pre-commit
install-pre-commit:
	pre-commit install --install-hooks
