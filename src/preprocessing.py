import pandas as pd
import numpy as np
import util as util

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from _preprocessing import _PreprocessingData

def load_dataset(config_data: dict, config: str) -> pd.DataFrame:
    # Load set of data
    X_train = util.pickle_load(config_data["train_set_eda"][0])
    y_train = util.pickle_load(config_data["train_set_eda"][1])
    
    X_valid = util.pickle_load(config_data["valid_set_eda"][0])
    y_valid = util.pickle_load(config_data["valid_set_eda"][1])
    
    X_test = util.pickle_load(config_data["test_set_eda"][0])
    y_test = util.pickle_load(config_data["test_set_eda"][1])
    
    dataset_train = pd.concat([X_train[config], y_train[config]], axis=1)
    dataset_valid = pd.concat([X_valid[config], y_valid[config]], axis=1)
    dataset_test = pd.concat([X_test[config], y_test[config]], axis=1)
    
    
    return dataset_train, dataset_valid, dataset_test

def _Concat_Preprocessing(data_filter) -> pd.DataFrame:
    le_encoder_filter = util.pickle_load(config_data['le_encoder_path_filter'])
    scaler_filter = util.pickle_load(config_data['scaler_filter'])
    
    X_filter, y_filter = preprocessor_._handling_data(data=data_filter,
                                                        encoding='label_encoder',
                                                        label_encod=util.pickle_load(config_data['le_encoder_path_filter']),
                                                        standard_scaler=util.pickle_load(config_data['scaler_filter'])
                                                        )
    
    return X_filter, y_filter

if __name__ == "__main__" :
    # 1. load Configuration file
    config_data = util.load_config()

    # 2. Load dataset
    train_set, valid_set, test_set = load_dataset(config_data)
    
    # 3. Load handling dataset
    preprocessor_ = _PreprocessingData()
    
    # 4. Load dataset
    X, y = _Concat_Preprocessing(data_filter=valid_set)
    
    
    
    