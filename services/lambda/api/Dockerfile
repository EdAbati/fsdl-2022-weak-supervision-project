FROM python:3.10.2-slim-bullseye as base

ARG AWS_REGION AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY

ENV AWS_REGION=${AWS_REGION} \
	AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} \
	AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}

WORKDIR /src

RUN python -m pip install boto3

COPY download_model_from_s3.py  /src

RUN python download_model_from_s3.py --destination /src/traced_model.pt


FROM public.ecr.aws/lambda/python:3.8 as api

COPY --from=base /src/traced_model.pt ${LAMBDA_TASK_ROOT}

COPY requirements.txt  .
RUN  python3.8 -m pip install -r requirements.txt --target "${LAMBDA_TASK_ROOT}" --no-cache-dir

COPY app.py ${LAMBDA_TASK_ROOT}

CMD ["app.lambda_handler"]
