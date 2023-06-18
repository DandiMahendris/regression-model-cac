import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from util import load_config
from util import pickle_dump
from util import print_debug

config_data = load_config()


class _PreprocessingData:
    """
    Handling raw dataset, \n
    Performed Imputer data, Label Encoding or OHE, and Standardization.
    """
    def __init__(self):
        pass
        
    def _split_xy(self, data:dict) -> pd.DataFrame:
        """Split dataset into Numerical (float64) and Categorical data (object).

        Parameters
        -------
        data : pandas.DataFrame
            train_set, valid_set included with label
        X : pandas.DataFrame
          Predictor array
        y : pandas.DataFrame
          label array

        Return 
        -------
        X_numeric : pandas.DataFrame
                        predictor for numeric only 
        X_categoric : pandas.DataFrame
                        predictor for categoric only
        """
        
        X_data = data.drop(columns = config_data['label'], axis=1)
        y_data = data[config_data['label']]
        
        self.X = X_data
        self.y = y_data
        
        numerical_col = X_data.select_dtypes('float64').columns.to_list()
        categorical_col = X_data.select_dtypes('object').columns.to_list()

        X_num = X_data[numerical_col]
        X_cat = X_data[categorical_col]

        return  X_num, X_cat
    
    # Perform sanity check
    def _imputer_Num(self, data, imputer=None):
        """
        Handling missing value for numeric if any. \n
        Using median to fill np.NaN by SimpleImputer() from Sklearn Function.

        Parameters
        ---------
        data : pandas.DataFrame
            Numeric (int64) dtype only

        Returns
        -------
        data_imputed : pandas.DataFrame
            Imputed numeric data
        """
        if imputer == None:
            imputer = SimpleImputer(missing_values=np.nan,
                                    strategy='median')
            imputer.fit(data)

        data_imputed_num = pd.DataFrame(imputer.transform(data),
                                    index = data.index,
                                    columns = data.columns)
        
        data_imputed_num = data_imputed_num.astype('int64')
        
        self.data_imputed_num = data_imputed_num
        
        return data_imputed_num

    def _imputer_Cat(self, data, imputer = None) -> pd.DataFrame:
        """
        Handling missing value for categorical data. \n
        Using 'most_frequent' strategy from SimpleImputer() of sklearn function

        Parameters
        --------
        data : pandas.DataFrame
            categorical (object) dtype only

        Return
        ----------
        imputed_cat : pandas.DataFrame
            imputed categorical data
        """
        if imputer == None:
            imputer = SimpleImputer(missing_values=np.nan,
                                    strategy='most_frequent')
            imputer.fit(data)

        data_imputed_cat = pd.DataFrame(imputer.transform(data),
                                    index=data.index,
                                    columns=data.columns
                                    )
        
        self.data_imputed_cat = data_imputed_cat
        
        return data_imputed_cat
    
    def _OHE_Cat(self, data, encoder_col = None, encoder = None) -> pd.DataFrame:
        """
        One Hot Encoding using OneHotEncoder() from sklearn.preprocessing \n
        handle_unknown : 'ignore' \n
        drop : 'if binary' \n
        This function for nominal or non-Ordinal categoric data \n
        If encoder_col and encoder == None, function will generate encoder from data.
        
        Parameters
        ------
        data : pandas.DataFrame
            categorical data non-Ordinal
        encoder_col : encoder.get_feature_names_out
        encoder : OneHotEncoder()

        Returns
        ------
        data_encoded : pd.DataFrame
            OHE encoded data
        encoder_cold
        encoder
        """

        if encoder == None:
            encoder = OneHotEncoder(handle_unknown= 'ignore',
                                    drop = 'if_binary')
            encoder.fit(data)
            encoder_col = encoder.get_feature_names_out(data.columns)

        data_encoded = encoder.transform(data).toarray()
        data_encoded = pd.DataFrame(data_encoded,
                                    index=data.index,
                                    columns=encoder_col)
        
        pickle_dump(encoder, config_data["ohe_path"])
        
        return data_encoded, encoder
    
    def _LE_cat(self, data, encoder = None) -> pd.DataFrame:
        """
        Label Encoder for Ordinal Categoric data using LabelEncoder() from sklearn.preprocessing function \n
        categories parameter is defined as of config file

        Parameters
        --------
        data : pandas.DataFrame
            Ordinal data only
        encoder : LabelEncoder()

        Returns
        ---------
        data_encoded : pandas.DataFrame
                Encoded ordinal data
        encoder : LabelEncoder()
        """
        
        if encoder == None:
            le_encoder = LabelEncoder()
                
        else:
            le_encoder = encoder
            
        for col in data.columns.to_list():
                data[col] = le_encoder.fit_transform(data[col])
        
        pickle_dump(le_encoder, config_data["le_encoder_path"])
        
        return data, le_encoder
    
    def _standardize_Data(self, data, scaler=None) -> pd.DataFrame:
        """
        Standarization or normalization of the predictor value using StandardScaler() from sklearn.preprocessing \n
        Standarization is use (x-mean)/std to get value range from -1 to 1 and gaussian distribution

        Paramters
        ----------
        data : pandas.DataFrame
            X_train data
        scaler : StandardScaler()

        Returns
        --------
        data_scaled : pandas.DataFrame
                standardized data
        scaler
        """
        
        if scaler == None:
            scaler = StandardScaler()
            scaler.fit(data)

        data_scaled = pd.DataFrame(scaler.transform(data),
                                index=data.index,
                                columns=data.columns)
        
        pickle_dump(scaler, config_data["scaler"])
        
        return data_scaled, scaler
    
    def _handling_data(self, data, encoding='label_encoder',
                       label_encod=None, encoder_ohe=None, standard_scaler=None,
                       imputer_num=None, imputer_cat=None,
                       method='None'):
        """
        Preprocessed data from dataset (X,y) into cleaned data

        Parameters
        ----------
        data : dataset with predictor and label (opt)
        split_Xy :  True, data will split into X and y
                    False, data will not splitted (data include predictor only)

        imputer :   True, data will be imputed
                    False, data will not imputed

        standardize : True, data will be normalize
                      False, data will not be normalize

        Returns
        ---------
        X_train : X_train clean
        y : label
        """
        print_debug(f"Split numeric and categoric data")
        num, cat = self._split_xy(data)
        
        print_debug("Perform imputer.")
        num = self._imputer_Num(data=num, imputer=imputer_num)
        cat = self._imputer_Cat(data=cat, imputer=imputer_cat)
        
        print_debug("Perform label encoding.")
        if encoding == 'label_encoder':
            X_train_le, encoder_le = self._LE_cat(cat, encoder=label_encod)
            X_train_ = pd.concat([num, X_train_le], axis=1)
            
        elif encoding == 'ohe':
            X_train_ohe, encoder_ohe_col, encoder_ohe = self._OHE_Cat(cat, encoder=encoder_ohe)
            X_train_ = pd.concat([num, X_train_ohe], axis=1)
        
        else:
            X_train_ohe, encoder_ohe_col, encoder_ohe = self._OHE_Cat(cat)
            X_train_le, encoder_le = self._LE_cat(data=cat) 
            
            X_train_concat = pd.concat([X_train_ohe, X_train_le], axis=1)
            X_train_ = pd.concat([num, X_train_concat], axis=1)
        
        print_debug("Perform Standardizing data.")
        X_train_ = X_train_.reindex(sorted(X_train_.columns), axis=1)

        X_clean, scaler_ = self._standardize_Data(X_train_, scaler=standard_scaler)

        if method == 'filter':
            pickle_dump(encoder_le, config_data["le_encoder_path_filter"])
            pickle_dump(scaler_, config_data["scaler_filter"])
        elif method == 'lasso':
            pickle_dump(encoder_le, config_data["le_encoder_path_lasso"])
            pickle_dump(scaler_, config_data["scaler_lasso"])
        elif method == 'random_forest':
            pickle_dump(encoder_le, config_data["le_encoder_path_rf"])
            pickle_dump(scaler_, config_data["scaler_rf"])
        else:
            pass
        
        return X_clean, self.y