from fastapi import FastAPI
from pydantic import BaseModel

import uvicorn
import pandas as pd
import util
import pipeline as pipeline
import preprocessing as preprocessing
import modelling as modelling

config_data = util.load_config()

# ohe_data = util.pickle_load(config_data["ohe_path_filter"])
le_label = util.pickle_load(config_data["le_encoder_path_filter"])
model_data = util.pickle_load(config_data["production_model_path"])

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
    data = pd.DataFrame([data]).reset_index(drop=True)
    
    # 2. Convert Dtype
    data.columns = config_data['api_predictor']
    
    # 3. Check range data
    try:
        pipeline.check_data(data, config_data)
    except AssertionError as ae:
        return {"res": [], "error_msg": str(ae)}
    
    # 4. Split data into predictor and label
    data = data[config_data["api_predictor"]].copy()
    
    # 5. Preprocessing data from label encoding and standardizing
    X, y = preprocessing._Concat_Preprocessing(data)
    
    # 6. Equalize the columns since OHE create 131 columns, with non existing value must have value = 0
    if len(data.columns) != 19:
        d_col = set(data.columns).symmetric_difference(set(X.columns))
        
        for col in d_col:
            data[col] = 0

    # 7. Predict the data
    y_pred = model_data["model_data"]["model_object"].predict(data)

    # 8. Return predicted data
    return {"res" : y_pred, "error_msg": ""}

if __name__ == "__main__":
    uvicorn.run("api:app", host = "0.0.0.0", port = 8080)

