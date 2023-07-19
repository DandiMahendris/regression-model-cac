import numpy as np
import pandas as pd
import util as util
import os
import copy

from tqdm import tqdm
from sklearn.model_selection import train_test_split

def read_raw_dataset(config: dict) -> pd.DataFrame:
    # Create variable to store raw dataset
    raw_dataset = pd.DataFrame()
    
    # Raw dataset dir
    raw_dataset_dir = config["raw_dataset_dir"]
    
    # Look and Load csv file
    for i in tqdm(os.listdir(raw_dataset_dir)):
        raw_dataset = pd.concat([pd.read_csv(raw_dataset_dir + i), raw_dataset])
        
    # Return raw dataset
    return raw_dataset

def convert_dataset_type(input_data):
    input_data.marital_status = input_data.marital_status.map(dict(S=0.0, M=1.0))
    input_data.gender = input_data.gender.map(dict(F=0.0, M=1.0))
    input_data.houseowner = input_data.houseowner.map(dict(N=0.0, Y=1.0))
    
    return input_data


def create_dataframe_new(data_list):
    # Create an empty dictionary to store column data
    column_data = {}

    # Iterate over the keys to initialize the empty lists for each column
    for key in data_list[0].keys():
        column_data[key] = []

    # Iterate over each row of data in the list
    for data in data_list:
        # Iterate over the keys and values in the data dictionary for each row
        for key, value in data.items():
            # Determine the data type for the column based on the value
            if isinstance(value, (int, float)):
                column_data[key].append(value)
            else:
                column_data[key].append(str(value))  # Convert non-numeric data to strings

    # Create the DataFrame using the column data
    dataset = pd.DataFrame(column_data)

    return dataset

def create_dataframe(data):
    # Create an empty dictionary to store column data
    column_data = {}

    # Iterate over the keys and values in the data dictionary
    for key, value in data.items():
        # Determine the data type for the column based on the value
        if isinstance(value, str):
            column_data[key] = [value]
        else:
            column_data[key] = [value]

    # Create the DataFrame using the column data
    dataset = pd.DataFrame(column_data)

    return dataset

def check_data(input_data, params, api = False):
    input_data = copy.deepcopy(input_data)
    params = copy.deepcopy(params)
    
    if not api:
        
        assert sorted(input_data.select_dtypes("float").columns.to_list()) == \
            sorted(params["float_predictor"]), "an error occurs in float column(s)."
        assert sorted(input_data.select_dtypes("object").columns.to_list()) == \
            sorted(params["object_predictor"]), "an error occurs in object column(s)."
            
        assert set(input_data.promotion_name).issubset(set(params["promotion_name"])), \
            "an error occurs in promotion_name range."
        assert set(input_data.sales_country).issubset(set(params["sales_country"])), \
            "an error occurs in sales_country range."
        assert set(input_data.occupation).issubset(set(params["occupation"])), \
            "an error occurs in occupation range."
        assert set(input_data.avg_yearly_income).issubset(set(params["avg_yearly_income"])), \
            "an error occurs in avg_yearly_income range."
        assert set(input_data.store_type).issubset(set(params["store_type"])), \
            "an error occurs in store_type range."
        assert set(input_data.store_city).issubset(set(params["store_city"])), \
            "an error occurs in store_city range."
        assert set(input_data.store_state).issubset(set(params["store_state"])), \
            "an error occurs in store_state range."
        assert set(input_data.media_type).issubset(set(params["media_type"])), \
            "an error occurs in media_type range."
            
        assert input_data.store_cost.between(params["store_cost"][0], params["store_cost"][1]).sum() == \
            len(input_data), "an error occurs in store_cost range."
        assert input_data.total_children.between(params["total_children"][0], params["total_children"][1]).sum() == \
            len(input_data), "an error occurs in total_children range."
        assert input_data.avg_cars_at_home.between(params["avg_cars_at_home"][0], params["avg_cars_at_home"][1]).sum() == \
            len(input_data), "an error occurs in avg_cars_at_home range."
        assert input_data.num_children_at_home.between(params["num_children_at_home"][0], params["num_children_at_home"][1]).sum() == \
            len(input_data), "an error occurs in num_children_at_home range."
        assert input_data.net_weight.between(params["net_weight"][0], params["net_weight"][1]).sum() == \
            len(input_data), "an error occurs in net_weight range."
        assert input_data.units_per_case.between(params["units_per_case"][0], params["units_per_case"][1]).sum() == \
            len(input_data), "an error occurs in units_per_case range."
        assert input_data.coffee_bar.between(params["coffee_bar"][0], params["coffee_bar"][1]).sum() == \
            len(input_data), "an error occurs in coffee_bar range."
        assert input_data.video_store.between(params["video_store"][0], params["video_store"][1]).sum() == \
            len(input_data), "an error occurs in video_store range."
        assert input_data.prepared_food.between(params["prepared_food"][0], params["prepared_food"][1]).sum() == \
            len(input_data), "an error occurs in prepared_food range."
        assert input_data.florist.between(params["florist"][0], params["florist"][1]).sum() == \
            len(input_data), "an error occurs in florist range."
        
        
    else:
        # In case checking data from api
        object_columns = params["object_predictor"]

        # Max column not used as predictor
        float_columns = params["float_predictor"]

        # Check data types
        assert input_data.select_dtypes("object").columns.to_list() == \
            object_columns, "an error occurs in object column(s)."
        assert input_data.select_dtypes("float").columns.to_list() == \
            float_columns, "an error occurs in float column(s)."
            
if __name__ == "__main__":
    # 1. Load configuration file
    config_data = util.load_config()

    # 2. Read all raw dataset
    raw_dataset = read_raw_dataset(config_data)

    # 3. Reset Index
    raw_dataset.reset_index(inplace=True, drop=True)

    # 4. Save raw dataset
    util.pickle_dump(raw_dataset, config_data["raw_dataset_path"])
    
    # 5. Handling data type
    raw_dataset  = convert_dataset_type(raw_dataset)
    
    # 7. Check data definition
    check_data(raw_dataset, config_data)
    
    # 8. Splitting data
    X = raw_dataset[config_data["predictor"]].copy()
    y = raw_dataset[config_data["label"]].copy()
    
    # 9. splitting train and test set with ratio 0.7:0.3 and do stratify splitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size = 0.3, 
                                                        random_state= 42)
    
    # 10. Splitting test and valid set with ratio 0.5:0.5 and do stratify splitting
    X_valid, X_test, y_valid, y_test = train_test_split(X_test, 
                                                        y_test, 
                                                        test_size = 0.5, 
                                                        random_state= 42)

    # 11. Save train, valid and test set
    util.pickle_dump(X_train, config_data["train_set_path"][0])
    util.pickle_dump(y_train, config_data["train_set_path"][1])

    util.pickle_dump(X_valid, config_data["valid_set_path"][0])
    util.pickle_dump(y_valid, config_data["valid_set_path"][1])

    util.pickle_dump(X_test, config_data["test_set_path"][0])
    util.pickle_dump(y_test, config_data["test_set_path"][1])
            
            
            
        
            
        