import sys
from dataclasses import dataclass
import os

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object


@dataclass
class DataTransformationconfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config  = DataTransformationconfig()

    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation initiated')

            # Define which columns should be ordinal-encoded and which should be scaled
            categorical_cols = ['workclass', 'education', 'marital_status', 'occupation',
       'relationship', 'race', 'sex']
            numerical_cols = ['age', 'education_num', 'capital_gain', 'hours_per_week']

            logging.info('Pipeline Initiated')

            ## Numerical Pipelines

            num_pipeline = Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())
                ]
            )

            ## Categorical Pipeline

            cat_pipeline = Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('one_hot_encoder', OneHotEncoder(sparse_output=False)),
                ('scaler',StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer([
                ('num_pipeline',num_pipeline,numerical_cols),
                ('cat_pipeline',cat_pipeline,categorical_cols)
            ])

            logging.info('Pipeline Completed')
            
            
            return preprocessor


        except Exception as e:
            logging.info('Error in Data Transformation')
            raise CustomException(e,sys)
        

    def initiate_data_transformation(self,train_path,test_path):
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            

            logging.info('Obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformation_object()

            income_map = {'<=50K':0,'>50K':1}

            train_df['income'] = train_df['income'].map(income_map)
            test_df['income'] = test_df['income'].map(income_map)
            logging.info('Mapping target(income) column')

            train_df.replace('?',np.nan,inplace=True)
            test_df.replace('?',np.nan,inplace=True)
            logging.info('Replacing ? with NaN')

            logging.info(f'Train Dataframe head: \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe head: \n{test_df.head().to_string()}')

            target_column_name = 'income'
            drop_columns = [target_column_name,'fnlwgt','capital_loss','native_country']

            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df = test_df[target_column_name]

            ## Transforming using preprocessor obj
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets.")
            

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            logging.info('Preprocessor pickle file saved')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )


        except Exception as e:
            logging.info('Exception occured in the initiate_data_transformation')
            raise CustomException(e,sys)

