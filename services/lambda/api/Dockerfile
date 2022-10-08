FROM public.ecr.aws/lambda/python:3.8

COPY app.py ${LAMBDA_TASK_ROOT}

COPY requirements.txt  .
RUN  python3.8 -m pip install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

CMD ["app.lambda_handler"]
