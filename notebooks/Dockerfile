FROM continuumio/miniconda3:latest

RUN mkdir -p /src/notebooks

WORKDIR /src/notebooks

COPY environment.yml requirements.txt /src/notebooks/

RUN conda env create -f environment.yml

# https://github.com/nlp-with-transformers/notebooks
# COPY notebooks /src/notebooks

# https://pythonspeed.com/articles/activate-conda-dockerfile/
RUN ["conda" ,"run", "--no-capture-output", "-n", "notebook-env", "python", "-c", "import sys;print('hello from docker');print(sys.version_info)"]

# https://stackoverflow.com/a/42873832/5819113
CMD ["conda" ,"run", "--no-capture-output", "-n", "notebook-env", "jupyter", "notebook", "--allow-root", "--ip", "0.0.0.0", "--port", "8888", "--notebook-dir=/src/notebooks"]
