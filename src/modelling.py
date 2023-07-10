import util as util
import numpy as np
import pandas as pd
import copy
import hashlib
import json

from sklearn.dummy import DummyRegressor
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, ridge_regression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

from sklearn.model_selection import KFold, cross_val_score

def load_train_clean(config_data: pd.DataFrame) -> pd.DataFrame:
        X = util.pickle_load(config_data["train_set_clean"][0])
        y = util.pickle_load(config_data["train_set_clean"][1])

        return X, y
    
def load_valid_clean(config_data: pd.DataFrame) -> pd.DataFrame:
        X = util.pickle_load(config_data["valid_set_clean"][0])
        y = util.pickle_load(config_data["valid_set_clean"][1])

        return X, y

def load_test_clean(config_data: pd.DataFrame) -> pd.DataFrame:
        X = util.pickle_load(config_data["test_set_clean"][0])
        y = util.pickle_load(config_data["test_set_clean"][1])

        return X, y

## Create training log function
def training_log_template() -> dict:
    # Debug message
    util.print_debug("creating training log template")

    # Template for training Log
    logger = {
        "model_name": [],
        "model_uid": [],
        "training_time": [],
        "training_date": [],
        "rmse": [],
        "r2_score": [],
    }

    # Debug message
    util.print_debug("Training log template created")

    return logger

def training_log_updater(current_log: dict, params: dict) -> list:
    # create copy of current log
    current_log = copy.deepcopy(current_log)

    # Path for training log file
    log_path = params["training_log_path"]

    # Try to load training log file
    try:
        with open(log_path, "r") as file:
            last_log = json.load(file)
        file.close()

    # If file not found create a new one
    except FileNotFoundError as fe:
        with open(log_path, "w") as file:
            file.write("[]")
        file.close()

        with open(log_path, "r") as file:
            last_log = json.load(file)
        file.close()

    # Add current log to previous log
    last_log.append(current_log)

    # Save updated log
    with open(log_path, "w") as file:
        json.dump(last_log, file)
        file.close()

    # Return log
    return last_log

## Create model object of ML model
def create_model_object(params: dict) -> list:
    # Debug message
    util.print_debug("Creating model objects.")

    # Create model objects
    baseline_knn = KNeighborsRegressor()
    baseline_dt = DecisionTreeRegressor()
    baseline_lr = LinearRegression()
    baseline_svr = SVR()
    baseline_rf = RandomForestRegressor()
    baseline_ada = AdaBoostRegressor()
    baseline_gr = GradientBoostingRegressor()
    baseline_xgb = XGBRegressor()

    # Create list of model
    list_of_model = [
        {"model_name": baseline_knn.__class__.__name__, "model_object": baseline_knn, "model_uid": ""},
        {"model_name": baseline_dt.__class__.__name__, "model_object": baseline_dt, "model_uid": ""},
        {"model_name": baseline_lr.__class__.__name__, "model_object": baseline_lr, "model_uid": ""},
        {"model_name": baseline_svr.__class__.__name__, "model_object": baseline_svr, "model_uid": ""},
        {"model_name": baseline_rf.__class__.__name__, "model_object": baseline_rf, "model_uid": ""},
        {"model_name": baseline_ada.__class__.__name__, "model_object": baseline_ada, "model_uid": ""},
        {"model_name": baseline_gr.__class__.__name__, "model_object": baseline_gr, "model_uid": ""},
        {"model_name": baseline_xgb.__class__.__name__, "model_object": baseline_xgb, "model_uid": ""},
    ]

    # Debug message
    util.print_debug("Models object created")

    return list_of_model

def train_eval(configuration_model: str, params: dict, hyperparams_model: list = None):

    # Variable to store trained models
    list_of_trained_model = dict()

    # Create training log template
    training_log = training_log_template()

    for config in X_train:
        # Debug message
        util.print_debug("Training model based on configuration data: {}".format(config_data))

        if hyperparams_model == None:
            list_of_model = create_model_object(params)
        else:
            list_of_model = copy.deepcopy(hyperparams_model)

        # Variable to store trained model
        trained_model = list()

        X_train_data = X_train[config]
        y_train_data = y_train[config]
        X_valid_data = X_valid[config]
        y_valid_data = y_valid[config]

        # Train each model by current dataset
        for model in list_of_model:
            # Debug message
            util.print_debug("Training model: {}".format(model["model_name"]))

            # Training
            training_time = util.time_stamp()
            model["model_object"].fit(X_train_data, y_train_data)
            training_time = (util.time_stamp() - training_time).total_seconds()

            # Debug message
            util.print_debug("Evaluating model: {}".format(model["model_name"]))

            # Evaluation
            y_predict = model["model_object"].predict(X_valid_data)
            rmse = mean_squared_error(y_valid_data, y_predict, squared=True)
            r2 = r2_score(y_valid_data, y_predict)

            # Debug message
            util.print_debug("Logging: {}".format(model["model_name"]))

            # Create UID
            uid = hashlib.md5(str(training_time).encode()).hexdigest()

            model["model_uid"] = uid

            # Create training log data
            training_log["model_name"].append("{}-{}-{}".format(configuration_model, config, model["model_name"]))
            training_log["model_uid"].append(uid)
            training_log["training_time"].append(training_time)
            training_log["training_date"].append(util.time_stamp())
            training_log["rmse"].append(rmse)
            training_log["r2_score"].append(r2)

            # Collenct current trained model
            trained_model.append(copy.deepcopy(model))

            # Debug Message
            util.print_debug("Model {} has been trained".format(model["model_name"]))

        # Collect current trained list of model
        list_of_trained_model[config] = copy.deepcopy(trained_model)

    # Debug message
    util.print_debug("All combination models and data has been trained.")


    return list_of_trained_model, training_log

def get_production_model(list_of_model, training_log, params):
    # Create copy list of model
    list_of_model = copy.deepcopy(list_of_model)
    
    # Debug message
    util.print_debug("Choosing model by metrics score.")

    # Create required predefined variabel
    curr_production_model = None
    prev_production_model = None
    production_model_log = None

    # Debug message
    util.print_debug("Converting training log type of data from dict to dataframe.")

    # Convert dictionary to pandas for easy operation
    training_log = pd.DataFrame(copy.deepcopy(training_log))

    # Debug message
    util.print_debug("Trying to load previous production model.")

    # Check if there is a previous production model
    try:
        prev_production_model = util.pickle_load(params["production_model_path"])
        util.print_debug("Previous production model loaded.")

    except FileNotFoundError as fe:
        util.print_debug("No previous production model detected, choosing best model only from current trained model.")

    # If previous production model detected:
    if prev_production_model != None:
        # Debug message
        util.print_debug("Loading validation data.")
        X_valid['filter'], y_valid['filter']
        
        # Debug message
        util.print_debug("Checking compatibilty previous production model's input with current train data's features.")

        # Check list features of previous production model and current dataset
        production_model_features = set(prev_production_model["model_data"]["model_object"].feature_names_in_)
        current_dataset_features = set(X_valid['filter'].columns)
        number_of_different_features = len((production_model_features - current_dataset_features) | (current_dataset_features - production_model_features))

        # If feature matched:
        if number_of_different_features == 0:
            # Debug message
            util.print_debug("Features compatible.")

            # Debug message
            util.print_debug("Reassesing previous model performance using current validation data.")

            # Re-predict previous production model to provide valid metrics compared to other current models
            y_pred = prev_production_model["model_data"]["model_object"].predict(X_valid['filter'])

            # Re-asses prediction result
            eval_res = mean_squared_error(y_valid['filter'], y_pred, squared = True)
            eval_r2 = r2_score(y_valid['filter'], y_pred)

            # Debug message
            util.print_debug("Assessing complete.")

            # Debug message
            util.print_debug("Storing new metrics data to previous model structure.")

            # Update their performance log
            prev_production_model["model_log"]["rmse"] = eval_res
            prev_production_model["model_log"]["r2_score"] = eval_r2

            # Debug message
            util.print_debug("Adding previous model data to current training log and list of model")

            # Added previous production model log to current logs to compere who has the greatest f1 score
            training_log = pd.concat([training_log, pd.DataFrame([prev_production_model["model_log"]])])

            # Added previous production model to current list of models to choose from if it has the greatest f1 score
            list_of_model["prev_production_model"] = [copy.deepcopy(prev_production_model["model_data"])]
        else:
            # To indicate that we are not using previous production model
            prev_production_model = None

            # Debug message
            util.print_debug("Different features between production model with current dataset is detected, ignoring production dataset.")

    # Debug message
    util.print_debug("Sorting training log by f1 macro avg and training time.")

    # Sort training log by f1 score macro avg and trining time
    best_model_log = training_log.sort_values(["rmse", "training_time"], ascending = [True, True]).iloc[0]
    
    # Debug message
    util.print_debug("Searching model data based on sorted training log.")

    # Get model object with greatest f1 score macro avg by using UID
    for configuration_data in list_of_model:
        for model_data in list_of_model[configuration_data]:
            if model_data["model_uid"] == best_model_log["model_uid"]:
                curr_production_model = dict()
                curr_production_model["model_data"] = copy.deepcopy(model_data)
                curr_production_model["model_log"] = copy.deepcopy(best_model_log.to_dict())
                curr_production_model["model_log"]["model_name"] = "Filter-{}".format(curr_production_model["model_data"]["model_name"])
                curr_production_model["model_log"]["training_date"] = str(curr_production_model["model_log"]["training_date"])
                production_model_log = training_log_updater(curr_production_model["model_log"], params)
                break
    
    # In case UID not found
    if curr_production_model == None:
        raise RuntimeError("The best model not found in your list of model.")
    
    # Debug message
    util.print_debug("Model chosen.")

    # Dump chosen production model
    util.pickle_dump(curr_production_model, params["production_model_path"])
    
    # Return current chosen production model, log of production models and current training log
    return curr_production_model, production_model_log, training_log

def eval_test(configuration_model: str, params: dict, hyperparams_model: list = None,
               X_t: pd.DataFrame = pd.DataFrame(), y_t: pd.DataFrame = pd.DataFrame()):

    # Variable to store trained models
    list_of_trained_model = dict()

    # Create training log template
    training_log = training_log_template()

    for config in X_train:
        # Debug message
        util.print_debug("Training model based on configuration data: {}".format(config))

        if hyperparams_model == None:
            list_of_model = create_model_object(params)
        else:
            list_of_model = copy.deepcopy(hyperparams_model)

        # Variable to store trained model
        trained_model = list()

        X_train_data = X_train[config]
        y_train_data = y_train[config]
        X_valid_data = X_t[config]
        y_valid_data = y_t[config]

        # Train each model by current dataset
        for model in list_of_model:
            # Debug message
            util.print_debug("Training model: {}".format(model["model_name"]))

            # Training
            training_time = util.time_stamp()
            model["model_object"].fit(X_train_data, y_train_data)
            training_time = (util.time_stamp() - training_time).total_seconds()

            # Debug message
            util.print_debug("Evaluating model: {}".format(model["model_name"]))

            # Evaluation
            y_predict = model["model_object"].predict(X_valid_data)
            rmse = mean_squared_error(y_valid_data, y_predict, squared=True)
            r2 = r2_score(y_valid_data, y_predict)

            # Debug message
            util.print_debug("Logging: {}".format(model["model_name"]))

            # Create UID
            uid = hashlib.md5(str(training_time).encode()).hexdigest()

            model["model_uid"] = uid

            # Create training log data
            training_log["model_name"].append("{}-{}-{}".format(configuration_model, config, model["model_name"]))
            training_log["model_uid"].append(uid)
            training_log["training_time"].append(training_time)
            training_log["training_date"].append(util.time_stamp())
            training_log["rmse"].append(rmse)
            training_log["r2_score"].append(r2)

            # Collenct current trained model
            trained_model.append(copy.deepcopy(model))

            # Debug Message
            util.print_debug("Model {} has been trained".format(model["model_name"]))

        # Collect current trained list of model
        list_of_trained_model[config] = copy.deepcopy(trained_model)

    # Debug message
    util.print_debug("All combination models and data has been trained.")


    return list_of_trained_model, training_log

def load_model():
    # 1. Load configuration file
    config_data = util.load_config()

    model_path = config_data["production_model_path"]

    model = util.pickle_load(model_path)

    model_data = model["model_data"]["model_object"]

    return model_data

if __name__ == "__main__":
    # 1. Load configuration file
    config_data = util.load_config()

    # 2. Load Dataset
    X_train, y_train = load_train_clean(config_data)
    X_valid, y_valid = load_valid_clean(config_data)
    X_test, y_test = load_test_clean(config_data)
    
    # 3. Training Dataset
    list_of_trained_model, training_log = train_eval("baseline", config_data)

    # 4. Save Best Model
    model, production_model_log, training_logs = get_production_model(list_of_trained_model, training_log, config_data)

    # 5. Create model object
    list_of_model = create_model_object(config_data)

    # 6. Cross Validation Score
    model_object = []
    model_name = []

    for model in list_of_model:
        model_object.append(model["model_object"])
        model_name.append(model["model_name"])

    cv = KFold(n_splits=5)

    for index, model in enumerate(model_object):
        cvs = cross_val_score(estimator=model, X=X_train['filter'], 
                            y=y_train['filter'], 
                            cv=cv, 
                            scoring='neg_root_mean_squared_error')
        mean = np.round(cvs.mean(), 3)
        std = np.round(cvs.std(), 3)
        print(f"cross validation score for the model {model_name[index]} is {mean} +/- {std}.")

    # 7. Check Model Performance
    list_of_test_model, testing_log = eval_test("baseline", config_data, 
                                             X_t=X_test, 
                                             y_t=y_test)
    
    # 8. Check Best Model on Validation test
    print(model['model_log'])

