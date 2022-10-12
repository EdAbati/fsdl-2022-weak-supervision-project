# Model API with AWS Lambda

This folder contains the code and supporting files needed to create a serverless model API.
It includes the following files and folders:

- `api/app.py` - Code for the application's Lambda function.
- `api/Dockerfile` - The Dockerfile to build the container image for local testing.
- `api/requirements.txt` - The pip requirements installed in the test container.

## Start API locally in a Docker container

Requirements:

- [Docker](https://docs.docker.com/get-docker/)

Create and start the API as a local docker container using `docker-compose`:

```bash
docker compose build lambda
docker compose -f docker-compose.yml -f docker-compose.override.yml up -d lambda
```

Testing with `curl`:

```bash
curl -POST "http://localhost:9000/2015-03-31/functions/function/invocations" -d '{"text": "A test sentence."}'
```

Testing with `python` and `requests`:

```python
import json

import requests

url = "http://localhost:9000/2015-03-31/functions/function/invocations"
headers = {"Content-Type": "application/json"}
payload = json.dumps({"text": 'A test sentence.'})
response = requests.post(url, data=payload, headers=headers)
```

Testing running the code in the [`notebooks/query_model_api.ipynb`](../../notebooks/query_model_api.ipynb) notebook.

## Deploy API to AWS Lambda

Requirements:

- [Docker](https://docs.docker.com/get-docker/)
- [An AWS Account](https://portal.aws.amazon.com/gp/aws/developer/registration/index.html?nc2=h_ct&src=header_signup)
- [AWS CLI](https://aws.amazon.com/cli/) installed and configured
- a `.env` file in the root of this project (based on `.env.sample`). Set the `AWS_ACCOUNT_ID` and `AWS_REGION` variables in this file.

1. Create a docker image and push to the ECR repository: `make deploy_to_aws_ecr`
2. Create a IAM role for the Lambda function: `make create_lambda_role`
3. Create the Lambda function: `make create_lambda_function`


## Contributing to the Lambda API service

If you want to contribute to this service you should:

1. Update one or more of the following files (based on how you want to change the API):
    - `app.py`: the logic of the API, the `lambda_handler` function is the entrypoint for the AWS Lambda function
    - `requirements.txt`: the python requirements of the app
    - `Dockerfile`: in case you want to change external dependency or the build of the image
2. Create and test API locally as described in '[Create API locally in a Docker container](/services/lambda/README.md#start-api-locally-in-a-docker-container)'
