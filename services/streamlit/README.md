# About

A [`streamlit`](streamlit.io) interface to use a news classifier accessible via an AWS Lambda function.

It consumes an environment variable `LAMBDA_URL` that points to an API endpoint that serves predictions from the AWS Lambda function.

## Screenshot

![screenshot](/docs/streamlit-UI-screenshot.png)

## Start UI locally

Requirements:

- [Docker](https://docs.docker.com/get-docker/)

1. Make sure the `lambda` API service is up on running or start it. More info [here](/services/lambda/README.md#start-api-locally-in-a-docker-container)
2. Use the `docker-compose` file (in the root of the project) to build and launch the `streamlit` service:

   ```bash
   docker compose build streamlit
   docker compose -f docker-compose.yml up -d streamlit
   ```

   pass the `override` config to mount the local directory to the container:

3. Access the UI at http://localhost:8501/

## Contributing to the Streamlit UI service

0. Navigate to the root of the project
1. Install `pre-commit` hooks with `make install-pre-commit`
2. Launch locally the `streamlit` and the `lambda` services using the `dev` config for `docker-compose`. This will mount the local directory to the container and it will allow you to change the local files and see changes in the UI straight away:

      ```bash
      docker compose build streamlit lambda
      docker compose -f docker-compose.yml -f docker-compose.override.yml up -d streamlit lambda
      ```
3. Make changes to the code and commit them to a new branch
4. Once you are happy with your changes, open a pull request
