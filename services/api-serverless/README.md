# Model API with AWS Lambda

This folder contains the code and supporting files needed to create a serveless model API.
It includes the following files and folders:

- `app/app.py` - Code for the application's Lambda function.
- `app/Dockerfile` - The Dockerfile to build the container image for local testing.
- `app/requirements.txt` - The pip requirements installed in the test container.


## Create API locally in a Docker container

Requirements:
- Docker

Start local docker container:
```bash
cd services/api-serverless/app
docker build -t news-model-api .
docker run -p 9000:8080 news-model-api
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
