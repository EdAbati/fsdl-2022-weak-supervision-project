FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    # software-properties-common \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /src/

WORKDIR /src/

# Build environment
RUN conda env create -f environment.yml
COPY requirements.txt /src/
RUN pip3 install -r requirements.txt

# copy application
COPY app /src/app
COPY .streamlit /root/.streamlit

EXPOSE 80

ENTRYPOINT ["streamlit"]
CMD [ "run", "/src/app/streamlit_app.py" ]
