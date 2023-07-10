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

def _Concat_(data_filter, data_lasso, data_rf):

    le_encoder_rf = util.pickle_load(config_data['le_encoder_path_rf'])
    scaler_rf = util.pickle_load(config_data['scaler_rf'])

    le_encoder_lasso = util.pickle_load(config_data['le_encoder_path_lasso'])
    scaler_lasso = util.pickle_load(config_data['scaler_lasso'])

    le_encoder_filter = util.pickle_load(config_data['le_encoder_path_filter'])
    scaler_filter = util.pickle_load(config_data['scaler_filter'])
    
    X_rf, y_rf = preprocessor_._handling_data(data=data_rf, 
                                             encoding='label_encoder',
                                             label_encod=util.pickle_load(config_data['le_encoder_path_rf']),
                                             standard_scaler=util.pickle_load(config_data['scaler_rf'])
                                             )

    X_lasso, y_lasso = preprocessor_._handling_data(data=data_lasso,
                                                    encoding='label_encoder',
                                                    label_encod=util.pickle_load(config_data['le_encoder_path_lasso']),
                                                    standard_scaler=util.pickle_load(config_data['scaler_lasso'])
                                                    )

    X_filter, y_filter = preprocessor_._handling_data(data=data_filter,
                                                        encoding='label_encoder',
                                                        label_encod=util.pickle_load(config_data['le_encoder_path_filter']),
                                                        standard_scaler=util.pickle_load(config_data['scaler_filter'])
                                                        )
    
    X = {
        "filter" : X_filter,
        "lasso" : X_lasso,
        "rf" : X_rf
    }
    
    y = {
        "filter" : y_filter,
        "lasso" : y_lasso,
        "rf" : y_rf
    }
    
    return X, y

def _Concat_Preprocessing(data_filter) -> pd.DataFrame:
    config_data = util.load_config()
    preprocessor_ = _PreprocessingData()

    le_encoder_filter = util.pickle_load(config_data['le_encoder_path_filter'])
    scaler_filter = util.pickle_load(config_data['scaler_filter'])
    
    X_filter = preprocessor_._handling_data(data=data_filter,
                                            encoding='label_encoder',
                                            label_encod=util.pickle_load(config_data['le_encoder_path_filter']),
                                            standard_scaler=util.pickle_load(config_data['scaler_filter']),
                                            y=False)
    
    return X_filter

if __name__ == "__main__" :
    # 1. load Configuration file
    config_data = util.load_config()

    # 2. Load dataset
    train_set_rf, valid_set_rf, test_set_rf = load_dataset(config_data, config='rf')
    train_set_filter, valid_set_filter, test_set_filter = load_dataset(config_data, config='filter')
    train_set_lasso, valid_set_lasso, test_set_lasso = load_dataset(config_data, config='lasso')
    
    # 3. Load function
    preprocessor_ = _PreprocessingData()

    # 4. Training dataset
    X_train_rf, y_train_rf = preprocessor_._handling_data(data=train_set_rf, 
                                             encoding='label_encoder',
                                             method='random_forest')

    X_train_lasso, y_train_lasso = preprocessor_._handling_data(data=train_set_lasso,
                                                                encoding='label_encoder',
                                                                method='lasso')

    X_train_filter, y_train_filter = preprocessor_._handling_data(data=train_set_filter,
                                                                encoding='label_encoder',
                                                                method='filter')

    # 5. Training dataset
    X_train = {
    "filter" : X_train_filter,
    "lasso" : X_train_lasso,
    "rf" : X_train_rf
    }

    y_train = {
        "filter" : y_train_filter,
        "lasso" : y_train_lasso,
        "rf" : y_train_rf
        }
    
    # 6. Validation dataset
    X_valid, y_valid = _Concat_Preprocessing(data_filter=valid_set_filter,
                                         data_lasso=valid_set_lasso,
                                         data_rf=valid_set_rf
                                         )
    
    # 7. Testing dataset
    X_test, y_test = _Concat_Preprocessing(data_filter = test_set_filter,
                                         data_lasso = test_set_lasso,
                                         data_rf = test_set_rf
                                         )
    
    # 8. Load dataset
    util.pickle_dump(X_train, config_data["train_set_clean"][0])
    util.pickle_dump(y_train, config_data["train_set_clean"][1])

    util.pickle_dump(X_valid, config_data["valid_set_clean"][0])
    util.pickle_dump(y_valid, config_data["valid_set_clean"][1])

    util.pickle_dump(X_test, config_data["test_set_clean"][0])
    util.pickle_dump(y_test, config_data["test_set_clean"][1])
    
    
    
    