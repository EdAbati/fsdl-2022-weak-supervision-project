from continuumio/miniconda3:latest


RUN mkdir -p /src/

WORKDIR /src/

COPY environment.yml /src/environment.yml

RUN conda env create -f environment.yml

# https://github.com/nlp-with-transformers/notebooks
COPY notebooks /src/notebooks

# https://pythonspeed.com/articles/activate-conda-dockerfile/
RUN ["conda" ,"run", "--no-capture-output", "-n", "fsdl", "python", "-c", "import sys;print('hello from docker');print(sys.version_info)"]

CMD ["conda" ,"run", "--no-capture-output", "-n", "fsdl", "jupyter-lab", "--allow-root", "--ip", "0.0.0.0", "--port", "8888", "--notebook-dir=/src/notebooks"]
