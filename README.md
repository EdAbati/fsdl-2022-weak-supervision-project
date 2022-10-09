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
- Optional, for AWS deployement:
    - [An AWS Account](https://portal.aws.amazon.com/gp/aws/developer/registration/index.html?nc2=h_ct&src=header_signup)
    - [AWS CLI](https://aws.amazon.com/cli/) installed and configured

### Setup project
Create a virtual environment using conda: `make conda-env-and-update`

## Usage

### Register model in W&B and convert to TorchScript

1. Add a registered model in the W&B model registry (guide [here](https://docs.wandb.ai/guides/models#model-registry-quickstart))
2. Run the command using the name in the registry as 'model_name':
    ```bash
    python cli/deploy.py register-and-convert-model --artifact-name="[entity]/[project]/[artifact_name]:[alias]" --model-name="[entity]/[project]/[model_name]"
    ```

This will register the model artifact in the registry and it will convert it in the TorchScript format.

### Start API


## Contributing

Create a dev environment running the following commands:

1. Create a virtual enviroment using conda: `make conda-env-and-update`
2. Install `pre-commit` hooks: `make install-pre-commit`
