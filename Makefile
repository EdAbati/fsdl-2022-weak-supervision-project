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
conda-env-and-update: ## Create and update a virtual environment using conda
	conda env update --prune -f conda-environment.yml
	@echo "!!! PLEASE ACTIVATE CONDA ENVIRONMENT !!!"

.PHONY: install-pre-commit
install-pre-commit: ## Install all pre-commit hooks
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

jupyter.model.train.default: ## train model inside the jupyter container with default arguments.
	@docker compose -f docker-compose.yml -f docker-compose.nvidia.yml exec -it jupyter fsdl-project-cli train

jupyter.model.train.help: ## Get help for training model inside the jupyter container.
	@docker compose -f docker-compose.yml -f docker-compose.nvidia.yml exec -it jupyter fsdl-project-cli train --help

dev.all.up: ## Start docker-compose in dev mode
	@docker compose -f docker-compose.yml -f docker-compose.override.yml up -d

dev.all.up.build: ## Start docker-compose in dev mode and --build flag to force rebuild
	@docker compose -f docker-compose.yml -f docker-compose.override.yml up -d --build

dev.all.ps: ## start docker-compose ps
	@docker compose -f docker-compose.yml -f docker-compose.override.yml ps

dev.all.down: ## stop docker-compose in dev mode
	@docker compose -f docker-compose.yml -f docker-compose.override.yml down

# AWS API related actions
# Based on the Makefile of https://github.com/caseyfitz/cookiecutter-disco-pie

ECR_URI = $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com
LAMBDA_FUNCTION_NAME = news-model-api
CONTAINER_NAME = fsdl-project/$(LAMBDA_FUNCTION_NAME)
LAMBDA_ROLE_NAME = $(LAMBDA_FUNCTION_NAME)-role
IMAGE_URI = $(ECR_URI)/$(CONTAINER_NAME)

.PHONY: build_aws_lambda_image
build_aws_lambda_image: ## Build AWS Lambda docker image locally
	docker build -t $(CONTAINER_NAME) . -f services/lambda/api/Dockerfile --platform=linux/amd64

.PHONY: run_local_aws_lambda
run_local_aws_lambda: ## Run AWS lambda docker image locally
	docker run -p 9000:8080 $(CONTAINER_NAME)

.PHONY: authenticate_aws_ecr
authenticate_aws_ecr: ## Authenticate to AWS ECR
	aws ecr get-login-password --region $(AWS_REGION) | docker login --username AWS --password-stdin $(ECR_URI)

.PHONY: create_aws_ecr_repository
create_aws_ecr_repository: authenticate_aws_ecr  ## Create a new AWS ECR repository
	aws ecr create-repository --repository-name $(LAMBDA_AND_CONTAINER_NAME) --image-scanning-configuration scanOnPush=true --image-tag-mutability MUTABLE

.PHONY: deploy_to_ecr
deploy_to_aws_ecr: build_aws_lambda_image authenticate_aws_ecr  ## Push image to AWS ECR repository
	docker tag $(CONTAINER_NAME):latest $(IMAGE_URI):latest
	docker push $(IMAGE_URI):latest

.PHONY: create_lambda_role
create_lambda_role: ## Create a IAM Role of the Lambda function
	aws iam create-role \
	--role-name $(LAMBDA_ROLE_NAME) \
	--assume-role-policy-document '{"Version": "2012-10-17","Statement": [{ "Effect": "Allow", "Principal": {"Service": "lambda.amazonaws.com"}, "Action": "sts:AssumeRole"}]}'

	aws iam attach-role-policy \
	--role-name $(LAMBDA_ROLE_NAME) \
	--policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

.PHONY: create_lambda_function
create_lambda_function: ## Create Lambda function based on the created Image and expose it with public url
	aws lambda create-function \
	--function-name $(LAMBDA_FUNCTION_NAME) \
	--region $(AWS_REGION) \
	--package-type Image \
	--code ImageUri=$(IMAGE_URI):latest \
	--role arn:aws:iam::$(AWS_ACCOUNT_ID):role/$(LAMBDA_ROLE_NAME)

	aws lambda create-function-url-config \
	--function-name $(LAMBDA_FUNCTION_NAME) \
	--auth-type NONE

	aws lambda add-permission \
	--function-name $(LAMBDA_FUNCTION_NAME) \
	--action lambda:InvokeFunction \
	--statement-id FunctionURLAllowPublicAccess \
	--principal * \
	--function-url-auth-type NONE
