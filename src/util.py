import yaml
import joblib
import pandas as pd
import os
import tqdm
from datetime import datetime

config_dir = "config/config.yaml"

def time_stamp() -> datetime:
    return datetime.now()

def load_config() -> dict:
    # Try to load yaml file
    try:
        with open(config_dir, "r") as file:
            config = yaml.safe_load(file)
    except:
        raise RuntimeError("parameter file not found in path")
    
    return config

def pickle_load(file_path: str):
    return joblib.load(file_path)

def pickle_dump(data, file_path: str) -> None:
    joblib.dump(data, file_path)
    
params = load_config()
PRINT_DEBUG = params["print_debug"]

def print_debug(message:str) -> None:
    if PRINT_DEBUG == True:
        print(time_stamp(), message)
    
def read_raw_data(params: dict) -> pd.DataFrame:
    raw_dataset = pd.DataFrame()
    
    raw_dataset_dir = params["raw_dataset_dir"]
    
    for i in tqdm(os.listdir(raw_dataset_dir)):
        raw_dataset = pd.concat([pd.read_csv(raw_dataset_dir + i), raw_dataset])
        
    return raw_dataset

def get_object_column(data) -> pd.DataFrame:
    lst_column = []
    for col in data.columns:
        if (data[col].dtype == 'O'):
            lst_column.append(col)
            
    return lst_column

def get_float_column(data) -> pd.DataFrame:
    lst_column = []
    for col in data.columns:
        if (data[col].dtype == 'float64'):
            lst_column.append(col)
            
    return lst_column