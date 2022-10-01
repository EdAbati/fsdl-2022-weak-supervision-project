FROM continuumio/miniconda3:latest

RUN mkdir -p /src/

WORKDIR /src/

COPY environment.yml /src/
RUN conda env create -f environment.yml

COPY requirements.txt /src/
RUN ["conda" ,"run", "--no-capture-output", "-n", "fsdl", "pip", "install", "-r", "requirements.txt"]

# https://github.com/nlp-with-transformers/notebooks
COPY notebooks /src/notebooks

# https://pythonspeed.com/articles/activate-conda-dockerfile/
RUN ["conda" ,"run", "--no-capture-output", "-n", "fsdl", "python", "-c", "import sys;print('hello from docker');print(sys.version_info)"]

COPY ./.jupyter/jupyter_server_config.py /root/.jupyter/jupyter_server_config.py

CMD ["conda" ,"run", "--no-capture-output", "-n", "fsdl", "jupyter-lab", "--allow-root", "--ip", "0.0.0.0", "--port", "8888", "--notebook-dir=/src/notebooks"]
