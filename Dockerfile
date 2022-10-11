FROM continuumio/miniconda3:latest

RUN mkdir -p /src/

WORKDIR /src/

COPY conda-environment.yml /src/
RUN conda env create -f conda-environment.yml

RUN apt-get update -y && apt-get install -y build-essential

COPY requirements.txt requirements.in /src/
RUN ["conda" ,"run", "--no-capture-output", "-n", "weak-active-supervision-project", "pip", "install", "-r", "requirements.txt", "--no-cache-dir"]

COPY setup.py pyproject.toml requirements.in /src/
COPY app /src/app
COPY notebooks /src/notebooks
# https://pythonspeed.com/articles/activate-conda-dockerfile/
RUN echo "conda activate weak-active-supervision-project" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

# Install library
RUN python3 -m pip install -e .

COPY services/jupyter/.jupyter/jupyter_server_config.py /root/.jupyter/jupyter_server_config.py

CMD ["conda" ,"run", "--no-capture-output", "-n", "weak-active-supervision-project", "jupyter-lab", "--allow-root", "--ip", "0.0.0.0", "--port", "8888", "--notebook-dir=/src/"]
