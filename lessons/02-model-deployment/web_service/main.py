# web_service/main.py
from fastapi.responses import HTMLResponse
#from lib.models import InputData
from lib.prediction_utils import load_pickle
from fastapi import FastAPI, Form, Query
from pydantic import BaseModel, StrictStr
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from typing import Dict
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import csr_matrix
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
#["PULocationID", "DOLocationID", "passenger_count"]
# Define a data model for the request body
# We're using StrictStr to ensure that the name is a string
# More information here: https://stackoverflow.com/questions/72263682/checking-input-data-types-in-pydantic
class Item(BaseModel):
    PULLocid: int
    location: int
    passenger_count: int

# Initiate the FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Serve static files (if any)
#app.mount("/static", StaticFiles(directory="web_service/static"), name="static")

# Home page
@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <html>
        <head>
            <title>Welcome to the page</title>
        </head>
        <body>
            <h1>Welcome to the page</h1>
            <p>Welcome to the page</p>
        </body>
    </html>
    """

# web_service/main.py (continued)
from fastapi import HTTPException

# Function to run inference
def run_inference(input_data: Item):
    try:
        model_path = "local_models/model.pkl"
        input_d = pd.DataFrame([input_data.dict()])
        
        numerical_cols = ["PULLocid", "passenger_count"]
        categorical_cols = ["location"]
        encoded_data = pd.get_dummies(input_d[categorical_cols], columns=categorical_cols)
        #print(encoded_data)
      
        # Concatenate processed numerical and categorical features
        input_data_processed = pd.concat([encoded_data, input_d[numerical_cols]], axis=1)
        # Quitar los títulos
        
        
        #data_dict: Dict = input_data.dict()
        #df = pd.DataFrame(data_dict)
        #numerical_dict = {key: value for key, value in data_dict.items() if isinstance(value, (int, float))}
        
        #vectorizer = DictVectorizer(sparse=True)
        #sparse_matrix = vectorizer.transform(numerical_dict)
        #csr_matrix_result = csr_matrix(sparse_matrix)
        def predict_duration(input_data, model: LinearRegression):
            return model.predict(input_data)


        print(input_data_processed.values)
        with open(model_path, 'rb') as model_file:
            pre_trained_model = pickle.load(model_file)
            #print(pre_trained_model)
            y = pre_trained_model.predict([True, 126, 1])
            print(y)
            return {y}
            
    
        
        """

        # Run inference using the provided input data
        #df = pd.DataFrame([x.dict() for x in input_data])
        #df = encode_categorical_cols(df)
        #dicts = df[CATEGORICAL_COLS].to_dict(orient="records")
        #x = dv.transform(dicts)
        # Convert the Pydantic BaseModel to a dictionary
        data_dict: Dict = input_data.dict()
        
        # Extract numerical values and create a dictionary for numerical features
        numerical_dict = {key: value for key, value in data_dict.items() if isinstance(value, (int, float))}
        
        # Use DictVectorizer to transform the dictionary into a sparse matrix
        vectorizer = DictVectorizer(sparse=True)
        sparse_matrix = vectorizer.transform(numerical_dict)
        
        # Convert the sparse matrix to csr_matrix
        csr_matrix_result = csr_matrix(sparse_matrix)
        y = load_pickle(csr_matrix_result, model_path)
        """
        

        return {"message": y}
    except Exception as e:
        # Handle any exceptions that might occur during inference
        raise HTTPException(status_code=500, detail=f"Error during inference: {str(e)}")

# Add the endpoint for running inference
@app.get("/run_inference", response_class=HTMLResponse)
async def run_inference_form():
    return """
    <html>
        <head>
            <title>Run Inference Form</title>
        </head>
        <body>
            <form method="post">
                <label>Loc Id:</label>
                <input type="number" name="PULLocid" required><br>
                <label>Location:</label>
                <input type="number" name="location"><br>
                <label>Passenger Count:</label>
                <input type="number" name="passenger_count" required><br>
                <br>
                <button title="hola" type="submit">Run Inference</button>
            </form>
        </body>
    </html>
    """

@app.post("/run_inference")
async def run_inference_endpoint(
    PULLocid: int = Form(...),
    location: int = Form(...),
    passenger_count: int = Form(...),
):
    try:
        
        input_data = Item(PULLocid=PULLocid, location=location, passenger_count=passenger_count)
        # Your logic here
        return {"message": f"Inference result for Loc Id {run_inference(input_data)}"}
    except Exception as e:
        return {"error": str(e)}
# Define an example endpoint with separate decorators for GET and POST
@app.get("/example")
@app.post("/example")
def example_endpoint():
    return {"message": "Hello, FastAPI!"}








