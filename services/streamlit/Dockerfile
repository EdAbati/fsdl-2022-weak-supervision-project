FROM continuumio/miniconda3:latest

RUN mkdir -p /src/

WORKDIR /src/

# Build environment
COPY environment.yml /src/
RUN conda env create -f environment.yml
COPY requirements.txt /src/
RUN ["conda" ,"run", "--no-capture-output", "-n", "weak-active-learner-streamlit", "pip", "install", "-r", "requirements.txt"]

# https://pythonspeed.com/articles/activate-conda-dockerfile/
RUN ["conda" ,"run", "--no-capture-output", "-n", "fsdl", "python", "-c", "import sys;print('hello from docker');print(sys.version_info)"]

# copy application
COPY app /src/app
COPY .streamlit /root/.streamlit

EXPOSE 80

ENTRYPOINT ["streamlit"]
CMD [ "run", "/src/app/streamlit_app.py" ]
