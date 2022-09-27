# Model API with AWS Lambda

This folder contains the code and supporting files needed to create a serveless model API.
It includes the following files and folders:

- `app/app.py` - Code for the application's Lambda function.
- `app/Dockerfile` - The Dockerfile to build the container image for local testing.
- `app/requirements.txt` - The pip requirements installed in the test container.


## Create API locally in a Docker container

Requirements:

- [Docker](https://docs.docker.com/get-docker/)

Create and start the API as a local docker container:

```bash
make build_aws_lambda_image
make run_local_aws_lambda
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

## Deploy API to AWS Lambda

Requirements:

- [Docker](https://docs.docker.com/get-docker/)
- [An AWS Account](https://portal.aws.amazon.com/gp/aws/developer/registration/index.html?nc2=h_ct&src=header_signup)
- [AWS CLI](https://aws.amazon.com/cli/) installed and configured
- a `.env` file in the root of this project (based on `.env.sample`). Set the `AWS_ACCOUNT_ID` and `AWS_REGION` variables in this file.


1. Create a docker image and push to the ECR repository: `make deploy_to_aws_ecr`
2. Create a IAM role for the Lambda function: `make create_lambda_role`
3. Create the Lambda function: `make create_lambda_function`
