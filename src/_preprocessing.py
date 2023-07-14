import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
# from . import util
import util

config_data = util.load_config()


class _Preprocessing_Data:
    """
    Handling raw dataset, \n
    Performed Imputer data, Label Encoding or OHE, and Standardization.
    """
    def __init__(self):
        pass

    def _split_numcat(self, data:dict) -> pd.DataFrame:
        """Split dataset without label into numerical and categorical.

        Parameters
        -------
        data : array-like of shape
            train_set without label

        Returns
        --------
        X_numerical : array-like of shape
            the training set containing only numerical features.
        
        X_category : array-like of shape
            the training set containing only categorical features.
        """
        numerical_col = data.select_dtypes('float64').columns.to_list()
        categorical_col = data.select_dtypes('object').columns.to_list()

        self.X_num = data[numerical_col]
        self.X_cat = data[categorical_col]

        return self.X_num, self.X_cat
        
    def _split_xy(self, data:dict) -> pd.DataFrame:
        """Split dataset into Numerical (float64) and Categorical data (object).

        Parameters
        -------
        data : array-like of shape
            train_set, valid_set included with label
        X : array-like of shape
          Predictor array
        y : array-like of shape
          label array

        Return 
        -------
        X_numeric : array-like of shape
                predictor for numeric only 
        X_categoric : array-like of shape
                predictor for categoric only
        """
        
        X = data.drop(columns = config_data['label'], axis=1)
        y = data[config_data['label']]
        
        self.X = X
        self.y = y
        
        numerical_col = X.select_dtypes('float64').columns.to_list()
        categorical_col = X.select_dtypes('object').columns.to_list()

        self.X_num = X[numerical_col]
        self.X_cat = X[categorical_col]

        return self.X_num, self.X_cat
    
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
        
        self.data_encoded = data_encoded
        self.encoder = encoder
        
        util.pickle_dump(encoder, config_data["ohe_path"])
        
        return self.data_encoded, self.encoder
    
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
        
        util.pickle_dump(le_encoder, config_data["le_encoder_path"])

        self.data_encoded = data
        self.encoder = le_encoder
        
        return self.data_encoded, self.encoder
    
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
        
        util.pickle_dump(scaler, config_data["scaler"])

        self.data_scaled = data_scaled
        self.scaler = scaler
        
        return self.data_scaled, self.scaler
    
    def _handling_data(self, 
                       data, 
                       encoding = 'Label_Encoding',
                       encoder = None, 
                       scaler = None,
                       imputer_num = None, 
                       imputer_cat = None,
                       config = "None", 
                       y = True):
        """
        Preprocessed data from dataset (X,y) into cleaned data

        Parameters
        ----------
        data : array-like of shape
            dataset with predictor and label (opt.)

        y : bool. (default = True)
            True, data will split into X and y (label)
            False, X (predictor) only

        imputer : SimpleImputer object. (default = None)

        scaler : StandarScaler object. (default = None)

        config : str (default = None)
            Type of config data to save imputer, encoder, and scaler.

            - 'filter' = filter data feature selection
            - 'lasso' = lasso data feature selection
            - 'random_forest' = random forest data feature selection

            If None, will saving none.

        Returns
        ---------
        X : array-like of shape
            encoded and scaled data predictor.

        y : array-like of shape (if y = True)
            label.
        """

        # --- Split into Numeric and Categoric Data --- #
        if y == False:
            util.print_debug("Splitting Numeric and Categoric Data...")

            self._split_numcat(data)

        elif y == True:
            self._split_xy(data)

        # --- Impute Missing Value --- #
        util.print_debug("Perform Imputer...")

        self.X_num = self._imputer_Num(self.X_num, imputer_num)
        self.X_cat = self._imputer_Cat(self.X_cat, imputer_cat)
        
        # --- Label Encoding Categoric data --- #

        # ---- Label Encoding --- #
        if encoding == 'Label_Encoding':
            util.print_debug("Perform Label Encoding...")

            self._LE_cat(self.X_cat, encoder)
            X_train_ = pd.concat([self.X_num, self.data_encoded], axis=1)
            
        # --- One Hot Encoding --- #
        elif encoding == 'One_Hot_Encoding':
            util.print_debug("Perform One Hot Encoding...")

            self._OHE_Cat(self.X_cat, encoder = encoder)
            X_train_ = pd.concat([self.X_num, self.data_encoded], axis=1)
        
        # --- Both --- #
        elif encoding == 'Both':
            util.print_debug("Perform Both Label Encoding and One Hot Encoding...")

            X_train_ohe, _, encoder_ohe = self._OHE_Cat(self.X_cat)
            X_train_le, encoder_le = self._LE_cat(self.X_cat) 
            
            X_train_concat = pd.concat([X_train_ohe, X_train_le], axis=1)
            X_train_ = pd.concat([self.X_num, X_train_concat], axis=1)

        else:
            raise TypeError("encoding type is not recognized. Should be Label_Encoding, One_Hot_Encoding, or Both.")
        
        # --- Standardize Data --- #
        util.print_debug("Perform Standardizing....")

        # Reindex data
        X_train_ = X_train_.reindex(sorted(X_train_.columns), axis=1)
        
        # Standardizing data
        self._standardize_Data(X_train_, scaler)

        util.print_debug("Data has been standardized.")

        # --- Dumping/Save data
        if config == 'filter':
            util.print_debug("Dumping encoder and scaler.")

            util.pickle_dump(self.encoder, config_data["le_encoder_path_filter"])
            util.pickle_dump(self.scaler, config_data["scaler_filter"])

        elif config == 'lasso':
            util.print_debug("Dumping encoder and scaler.")

            util.pickle_dump(self.encoder, config_data["le_encoder_path_lasso"])
            util.pickle_dump(self.scaler, config_data["scaler_lasso"])

        elif config == 'random_forest':
            util.print_debug("Dumping encoder and scaler.")

            util.pickle_dump(self.encoder, config_data["le_encoder_path_rf"])
            util.pickle_dump(self.scaler, config_data["scaler_rf"])

        else:
            pass

        util.print_debug("Returned scaled data.")
        util.print_debug("="*40)

        if y == True:
            return self.data_scaled, self.y
        else:
            return self.data_scaled