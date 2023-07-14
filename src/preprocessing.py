import pandas as pd
import numpy as np
import util as util

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from _preprocessing import _Preprocessing_Data

def load_dataset(config_data: dict, config: str) -> pd.DataFrame:
    """
    After conducting exploratory data analysis (EDA), 
    load the dataset consisting of the train set, valid set, and test set.
    
    Parameters
    -----
    config : str
        - 'rf' = Random Forest features selection set
        - 'filter' = Filter features selection set
        - 'lasso' = Lasso featrues selection set

    Returns
    -----
    train_set : array-like of shape
        training data set
    
    valid_set : array-like of shape
        validation data set

    test_set : array-like of shape
        testing data set
    """
    
    # ---
    config_data = util.load_config()

    # Load set of data
    X_train = util.pickle_load(config_data["train_set_eda"][0])
    y_train = util.pickle_load(config_data["train_set_eda"][1])
    
    X_valid = util.pickle_load(config_data["valid_set_eda"][0])
    y_valid = util.pickle_load(config_data["valid_set_eda"][1])
    
    X_test = util.pickle_load(config_data["test_set_eda"][0])
    y_test = util.pickle_load(config_data["test_set_eda"][1])
    
    train_set = pd.concat([X_train[config], y_train[config]], axis=1)
    valid_set = pd.concat([X_valid[config], y_valid[config]], axis=1)
    test_set = pd.concat([X_test[config], y_test[config]], axis=1)
    
    
    return train_set, valid_set, test_set

def _Concat_Preprocessing(data_filter, data_lasso, data_rf, train = False):
    """Preprocessing data filter, data lasso, and data random forest.

    Parameters
    ----
    data_filter : array-like of shape
        filter dataset

    data_lasso : array-like of shape
        lasso dataset

    data_rf : array-like of shape
        random forest dataset

    train : bool (default = False)
        - True = dumping or save imputer, encoder and scaler.
        - False = non dumping and use fitted imputer, encoder, and scaler. 

    Returns
    -----
    X : dict
        Return a dictionary that stores the predictor (X) for every type of configuration.
    y : dict
        Return a dictionary that stores the y (label) for every type of configuration.
    
    """

    preprocessor_ = _Preprocessing_Data()
    
    if train == True:
        
        # Random Forest Feature Selection
        X_rf, y_rf = preprocessor_._handling_data(data = data_rf, 
                                                encoding = 'Label_Encoding',
                                                config = 'filter',
                                                train = train
                                                )

        # Lasso Feature Selection
        X_lasso, y_lasso = preprocessor_._handling_data(data = data_lasso,
                                                        encoding = 'Label_Encoding',
                                                        config = 'lasso',
                                                        train = train
                                                        )

        # Filter Feature Selection
        X_filter, y_filter = preprocessor_._handling_data(data = data_filter,
                                                            encoding = 'Label_Encoding',
                                                            config = 'random_forest',
                                                            train = train
                                                            )
        
    elif train == False:

        # Random Forest Feature Selection
        X_rf, y_rf = preprocessor_._handling_data(data = data_rf, 
                                                encoding = 'Label_Encoding',
                                                config = 'random_forest',
                                                encoder = util.pickle_load(config_data['le_encoder_path_rf']),
                                                scaler = util.pickle_load(config_data['scaler_rf']),
                                                train = train
                                                )

        # Lasso Feature Selection
        X_lasso, y_lasso = preprocessor_._handling_data(data = data_lasso,
                                                        encoding = 'Label_Encoding',
                                                        config = 'lasso',
                                                        encoder = util.pickle_load(config_data['le_encoder_path_lasso']),
                                                        scaler = util.pickle_load(config_data['scaler_lasso']),
                                                        train = train
                                                        )

        # Filter Feature Selection
        X_filter, y_filter = preprocessor_._handling_data(data = data_filter,
                                                            encoding = 'Label_Encoding',
                                                            config = 'random_forest',
                                                            encoder = util.pickle_load(config_data['le_encoder_path_filter']),
                                                            scaler = util.pickle_load(config_data['scaler_filter']),
                                                            train = train
                                                            )
        
    else:
        raise TypeError("train status is not recognize, should be True of False.")
        
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

def _Preprocessing_Test(data) -> pd.DataFrame:

    config_data = util.load_config()
    
    preprocessor_ = _Preprocessing_Data()
    
    X = preprocessor_._handling_data(
                                    data = data,
                                    encoding = 'Label_Encoding',
                                    encoder = util.pickle_load(config_data['le_encoder_path_filter']),
                                    scaler = util.pickle_load(config_data['scaler_filter']),
                                    y = False
                                    )
    
    return X

if __name__ == "__main__" :
    # 1. load Configuration file
    config_data = util.load_config()

    # 2. Load dataset
    train_set_rf, valid_set_rf, test_set_rf = load_dataset(config_data, config='rf')
    train_set_filter, valid_set_filter, test_set_filter = load_dataset(config_data, config='filter')
    train_set_lasso, valid_set_lasso, test_set_lasso = load_dataset(config_data, config='lasso')
    
    # 3. Preprocessing data
    X_train, y_train = _Concat_Preprocessing(data_filter = train_set_filter, 
                                             data_lasso = train_set_lasso, 
                                             data_rf = train_set_rf, 
                                             train = True
                                             )

    
    # 6. Validation dataset
    X_valid, y_valid = _Concat_Preprocessing(data_filter = valid_set_filter, 
                                             data_lasso = valid_set_lasso, 
                                             data_rf = valid_set_rf
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
    
    
    
    