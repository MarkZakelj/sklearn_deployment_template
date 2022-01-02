FROM python:3.9

WORKDIR /app

COPY ./requirements.txt /app

RUN pip install --no-cache-dir  -r requirements.txt

# load model and fastapi files
COPY ./rf_model.joblib /app
COPY ./main.py /app

COPY ./__init__.py /app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0"]
