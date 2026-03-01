import os
import pandas as pd
import joblib  
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from IPL.entity.config_entity import DataTransformationConfig
from IPL import logger  

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config    

    def get_data_transformation_object(self):

        categorical_columns = ['batting_team', 'bowling_team', 'city']
        
        preprocessor = ColumnTransformer([
            ('onehot', OneHotEncoder(sparse_output=False, drop='first'), categorical_columns)
        ], remainder='passthrough')
        
        return preprocessor

    def initiate_data_transformation(self):
        try:

            df = pd.read_csv(self.config.data_path)
            logger.info("Read raw data successfully")
            
            X = df.drop(columns=['result'], axis=1)
            y = df['result']

           
            preprocessing_obj = self.get_data_transformation_object()
            
            X_encoded = preprocessing_obj.fit_transform(X)
            
            
            transformer_path = os.path.join(self.config.root_dir, "transformer.joblib")
            joblib.dump(preprocessing_obj, transformer_path)
            logger.info(f"Saved preprocessing object to {transformer_path}")
            
            column_names = preprocessing_obj.get_feature_names_out()
            X_df = pd.DataFrame(X_encoded, columns=column_names)

            final_df = pd.concat([X_df, y.reset_index(drop=True)], axis=1)

     
            train, test = train_test_split(final_df, test_size=0.2)
            
            train_path = os.path.join(self.config.root_dir, "train.csv")
            test_path = os.path.join(self.config.root_dir, "test.csv")
            
            train.to_csv(train_path, index=False)
            test.to_csv(test_path, index=False)
            
            logger.info("Split data into numeric train and test sets")
            
            return (train_path, test_path)

        except Exception as e:
            raise e