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

   pass the `dev` config to mount the local directory to the container:

3. Access the UI at http://localhost:8501/

## Contributing to the Streamlit UI service

If you want to contribute to this service you should:

0. Create and activate a dev environment with `conda`:
   1. Run `make conda-env-and-update`.
   2. **Activate** the environment
   3. Run `make pip-tools` to install and update requirements.
1. Launch locally the `streamlit` and the `lambda` services using the `dev` config for `docker-compose`. This will mount the local directory to the container and it will allow you to change the local files and see changes in the UI straight away:

      ```bash
      docker compose build streamlit lambda
      docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d streamlit lambda
      ```
2. Make changes to the app code in `app/streamlit_app.py` and commit them to a new branch
3. Once you are happy with your changes, open a pull request

### Update app requirements

Requirement are managed by `pip-tools` and compiled inside a virtual environment.

1. Create and activate a dev environment with `conda`:`make conda-env-and-update`.
2. **Activate** the environment
3. Change the requirements in `requirements.in`
4. Run `make pip-tools` to requirements. This will create a `requirements.txt` that is used by the Docker container.
