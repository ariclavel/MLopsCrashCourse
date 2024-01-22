# web_service/lib/prediction_utils.py
from sklearn.linear_model import LinearRegression
from scipy.sparse import csr_matrix
from typing import Any
import numpy as np
import joblib
import pickle
from scipy.sparse import csr_matrix
from sklearn.linear_model import LinearRegression


def load_pickle(path: str):
    with open(path, "rb") as f:
        loaded_obj = pickle.load(f)
    return loaded_obj


def save_pickle(path: str, obj: Any):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def predict_duration(input_data, model: LinearRegression):
    return model.predict(input_data,check_input=False)


def load_pickle(input_data, path: str):
    with open(path, 'rb') as model_file:
        pre_trained_model = pickle.load(model_file)
        return predict_duration(input_data,pre_trained_model)





