from fastapi import FastAPI
from typing import List
from pydantic import BaseModel, Field
from util import print_debug

import uvicorn
import pandas as pd
import util
import pipeline as pipeline
import preprocessing as preprocessing
import modelling as modelling

config_data = util.load_config()

# ohe_data = util.pickle_load(config_data["ohe_path_filter"])
le_label = util.pickle_load(config_data["le_encoder_path_filter"])
# model_data = util.pickle_load(config_data["production_model_path"])

class api_data(BaseModel):
    store_cost : float
    total_children : float
    avg_cars_at_home : float
    num_children_at_home : float
    net_weight : float
    units_per_case : float
    coffee_bar : float
    video_store : float
    prepared_food : float
    florist : float
    promotion_name : str
    sales_country : str
    occupation : str
    avg_yearly_income : str
    store_type : str
    store_city : str
    store_state : str
    media_type : str


app = FastAPI()

@app.get("/")
def home():
    return "Hello, FastAPI is up!"

@app.post("/predict/")
def predict(data: api_data):
    # 0. Load config
    config_data = util.load_config()

    # 1. Convert data api to dataframe
    data_dict = data.dict()
    dataset = pipeline.create_dataframe(data_dict)

    # 2. Convert Dtype
    dataset.columns = config_data['api_predictor']

    # 3. Check range data
    try:
        pipeline.check_data(dataset, config_data)
    except AssertionError as ae:
        return {"res": [], "error_msg": str(ae)}
    
    print_debug("dataset has been validated.")

    print(dataset.head(5))
    
    # 4. Preprocessing data from label encoding and standardizing
    X = preprocessing._Preprocessing_Test(dataset)

    print_debug("dataset has been preprocessed.")

    # Sort index to be matched with model
    X = X.reindex(sorted(X.columns), axis=1)

    print_debug("dataset has been reindex.")
    
    # 5. Equalize the columns since OHE create 131 columns, with non-existing value must have value = 0
    if len(dataset.columns) != 18:
        d_col = set(dataset.columns).symmetric_difference(set(X.columns))
        
        for col in d_col:
            dataset[col] = 0

    # 6. Load and initialize the model for each prediction
    model = modelling.load_model()  # Replace with your model loading code
    
    
    # 7. Predict the data
    print_debug("predict data.")
    
    y_pred = model.predict(X)
    y_pred = y_pred.tolist()

    print_debug("returned predicted data.")
    print_debug(f"Customer Acquisition Cost is {y_pred}")
    
    # 8. Return predicted data
    return {"res": y_pred, "error_msg": ""}

if __name__ == "__main__":
    uvicorn.run("api:app", host = "0.0.0.0", port = 8080)

