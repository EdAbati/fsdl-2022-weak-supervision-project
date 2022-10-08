FROM public.ecr.aws/lambda/python:3.8

RUN ["yum", "install", "-y", "tar", "vim"]

COPY app.py ${LAMBDA_TASK_ROOT}
ARG PYTHONPATH=${LAMBDA_TASK_ROOT}
ENV PYTHONPATH=${PYTHONPATH}


COPY requirements.txt  .
RUN  python3.8 -m pip install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

CMD ["app.lambda_handler"]
