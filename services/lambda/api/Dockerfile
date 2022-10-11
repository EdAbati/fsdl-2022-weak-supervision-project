FROM public.ecr.aws/lambda/python:3.8

COPY /models/traced_model.pt ${LAMBDA_TASK_ROOT}

COPY services/lambda/api/requirements.txt .
RUN  python3.8 -m pip install -r requirements.txt --target "${LAMBDA_TASK_ROOT}" --no-cache-dir

COPY services/lambda/api/app.py ${LAMBDA_TASK_ROOT}

CMD ["app.lambda_handler"]
