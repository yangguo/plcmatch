FROM python:3.9

WORKDIR /app

COPY requirements.txt ./requirements.txt

RUN pip install --upgrade pip setuptools wheel

RUN pip install -r requirements.txt 

# download the spacy model
RUN python -m spacy download zh_core_web_lg

EXPOSE 8501

COPY . /app

ENTRYPOINT ["streamlit", "run"]

CMD ["app.py"]