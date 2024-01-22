from pydantic import BaseModel
from scipy.sparse import csr_matrix

class InputData(BaseModel):
    feature_matrix: csr_matrix  # Assuming the input data is a csr_matrix
    model_path: str

"""
# Example of using the InputData model in a FastAPI endpoint
from fastapi import FastAPI

app = FastAPI()

@app.post("/predict")
def predict_duration(input_data: InputData):
    # Load the model from the specified path using the load_pickle function
    model = load_pickle(input_data.model_path)

    # Assuming the model is a LinearRegression object
    prediction_result = model.predict(input_data.feature_matrix)

    return {"prediction_result": prediction_result}
"""


