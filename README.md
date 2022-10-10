# FSDL 2022: Weak Supervision Project

[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/EdAbati/fsdl-2022-weak-supervision-project/main.svg)](https://results.pre-commit.ci/latest/github/EdAbati/fsdl-2022-weak-supervision-project/main)

Authors:

- ...

## Description

...

## Requirements

- [Docker](https://docs.docker.com/get-docker/)
- [`conda`](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
- a `.env` file in the root of this project (based on `.env.sample`)
- Optional, for AWS deployment:
  - [An AWS Account](https://portal.aws.amazon.com/gp/aws/developer/registration/index.html?nc2=h_ct&src=header_signup)
  - [AWS CLI](https://aws.amazon.com/cli/) installed and configured

### Setup project

Create a virtual environment using `conda`: `make conda-env-and-update`

## Usage

### Register model in W&B and convert to TorchScript

1. Add a registered model in the W&B model registry (guide [here](https://docs.wandb.ai/guides/models#model-registry-quickstart))
2. Run the command using the name in the registry as 'model_name':

   ```bash
   python cli/deploy.py register-and-convert-model --artifact-name="[entity]/[project]/[artifact_name]:[alias]" --model-name="[entity]/[project]/[model_name]"
   ```

This will register the model artifact in the registry and it will convert it into the TorchScript format.

## Contributing

Create a dev environment running the following commands:

1. Create a virtual environment using `conda`: `make conda-env-and-update`
2. Install `pre-commit` hooks: `make install-pre-commit`

## Using docker-compose

This project has the following services

1. `proxy`: a `traefik` reverse proxy.
1. `restapi`: a FastAPI application, currently not being used, in favor of a lambda function
1. `jupyter`: a Jupyter notebook server with support for the NVIDIA A100-SXM4-40GB GPU (CUDA 11.6) and contains scripts for training and evaluating models
1. `streamlit`: a Streamlit application for displaying a user interface
1. `rubrix`: a Rubrix server for annotating data
1. `elastic`: an Elasticsearch server for storing data from rubrix
1. `kibana`: a Kibana server for visualizing data from Elasticsearch
1. `lambda`: a lambda function for serving predictions

these can be launched using docker compose, and its configuration is shared across many `yml` files that can by chained with the `-f` command. Read more about this in the [docker-compose documentation](https://docs.docker.com/compose/extends/). You will need to provide a `.env` file with variables listed in `.env.sample` for this deployment to work.

In particular, the following files are used:

- `docker-compose.yml` (`base`) is used to run the project in a container. It supports docker swarm for production but it has not been tested yet.
- `docker-compose.override.yml` (`override`) is used to run the project in a container with a volume mounted to the host machine. It also has `traefik` labels for the proxy service that work behind a reverse proxy. It also
- `docker-compose.nvidia.yml` (`nvidia`) extends the `jupyter` container and adds support for GPU usage.

These can be launched using `make`, e.g. `make dev.all.up` will launch all the services combining the `base` and `override` configs. The `Makefile` contains a list of all the commands.
