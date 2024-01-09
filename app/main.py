import uvicorn
from fastapi import FastAPI
import pandas as pd 
import joblib
import numpy as np
from pydantic import BaseModel
from data_and_model_tests import load_trained_model

# create an instance of fastapi

app= FastAPI()


class Inputs(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    
    
@app.get("/")
def read_root():
    return {"message": "Hello, this is your FastAPI endpoint!"}

@app.post('/predict')
def predict_species(data:Inputs):
    data = data.dict()
    sepal_length = data['sepal_length']
    sepal_width = data['sepal_width']
    petal_length = data['petal_length']
    petal_width = data['petal_width']
    
    # load the trained model from data_and_model_tests
    model = load_trained_model()
    
    prediction = model.predict([[sepal_length,sepal_width, petal_length,petal_width]])
    
    return {
        'prediction': prediction[0]
    }

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8080)