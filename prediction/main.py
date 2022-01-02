from typing import Optional
from fastapi import FastAPI
from joblib import load
from pydantic import BaseModel
from enum import IntEnum
import numpy as np

class TargetClass(IntEnum):
    iris_setosa = 0
    iris_versicolour = 1
    iris_virginica = 2


class UserRequestIn(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class PredictionOut(BaseModel):
    target_class: TargetClass


app = FastAPI()
model = load('rf_model.joblib')

@app.get("/")
def read_root():
    return {"main": "Hello world"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {"main": "item getter", "item_id": item_id, "q": q}

@app.post("/inference", response_model=PredictionOut)
def model_inference(input: UserRequestIn):
    sl = input.sepal_length
    sw = input.sepal_width
    pl = input.petal_length
    pw = input.petal_width
    model_input = np.array([[sl, sw, pl, pw]])
    prediction = {
        'target_class': TargetClass(model.predict(model_input))
    }
    return prediction
    