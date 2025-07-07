import sys
import os

import pandas as pd
import numpy as np

from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler #LabelEncoder
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.linear_model import LogisticRegression, SGDClassifier
# from sklearn.metrics import classification_report, confusion_matrix
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path:str = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
        def __init__(self):
            self.data_transformation_config=DataTransformationConfig()
            print("DataTransformation initialized.")

        def get_data_transformer_object(self):
            '''
            This function is responsible for data transformation
            '''
            print("Returning data transformer object.")
         
            try:
                numerical_columns = ["age","hypertension","heart_disease","bmi","HbA1c_level","blood_glucose_level"]
                categorical_columns = ["gender","smoking_history"]
                

                num_pipeline= Pipeline(
                    steps=[
                        ("imputer",SimpleImputer(strategy="median")),
                        ("scaler",RobustScaler()),
                        
                    ])

                cat_pipeline= Pipeline(
                    steps=[
                        ("imputer",SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
                    ])

                logging.info("Numerical columns robust scaling completed")

                logging.info("Categorical columns encoding completed")

                preprocessor=ColumnTransformer([
                        ("num_pipeline",num_pipeline,numerical_columns),
                        ("cat_pipeline",cat_pipeline,categorical_columns)
                    ])

                return preprocessor
                
            except Exception as e:
                raise CustomException(e,sys)
            
        def initiate_data_transformation(self,train_path,test_path):

            try:
                train_df=pd.read_csv(train_path)
                test_df=pd.read_csv(test_path)

                #train_df.columns = train_df.columns.str.strip()
                #test_df.columns = test_df.columns.str.strip()

                #print("Train DF columns:", train_df.columns.tolist())
                #print("Test DF columns:", test_df.columns.tolist())

                logging.info("Read train and test data completed")
                preprocessing_obj=self.get_data_transformer_object()

                
                target_column_name= "diabetes" 

                input_feature_train_df=train_df.drop(columns=[target_column_name], axis=1)
                target_feature_train_df=train_df[target_column_name]

                input_feature_test_df=test_df.drop(columns=[target_column_name], axis=1)
                target_feature_test_df=test_df[target_column_name]

                input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
                input_feature_test_arr=preprocessing_obj.fit_transform(input_feature_test_df)

                train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
                test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

                
                save_object(

                    file_path=self.data_transformation_config.preprocessor_obj_file_path,
                    obj=preprocessing_obj
                )

                logging.info(f"Saved preprocessing object.")

                return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

            except Exception as e:
                raise CustomException(e,sys)
            

if __name__ == "__main__":
    try:
        # Provide the correct paths to your train and test CSV files
        train_path = os.path.join('artifacts', 'train.csv')
        test_path = os.path.join('artifacts', 'test.csv')

        transformer = DataTransformation()
        transformer.initiate_data_transformation(train_path, test_path)

        print("Data transformation completed successfully.")

    except Exception as e:
        print(f"Error during transformation: {e}")
