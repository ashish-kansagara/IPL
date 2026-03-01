import os
import pandas as pd
import joblib  # Required for saving the transformer artifact
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from IPL.entity.config_entity import DataTransformationConfig
from IPL import logger  # Assuming you have a logger defined in your project

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config    

    def get_data_transformation_object(self):

        categorical_columns = ['batting_team', 'bowling_team', 'city']
        
        # sparse_output=False ensures we get a dense array for the DataFrame
        preprocessor = ColumnTransformer([
            ('onehot', OneHotEncoder(sparse_output=False, drop='first'), categorical_columns)
        ], remainder='passthrough')
        
        return preprocessor

    def initiate_data_transformation(self):
        try:
            # 1. Load Data from the path specified in config
            df = pd.read_csv(self.config.data_path)
            logger.info("Read raw data successfully")
            
            # 2. Separate Features and Target
            X = df.drop(columns=['result'], axis=1)
            y = df['result']

            # 3. Fit and Apply Encoding
            preprocessing_obj = self.get_data_transformation_object()
            
            # We fit on X to learn the categories (Teams and Cities)
            X_encoded = preprocessing_obj.fit_transform(X)
            
            # 4. SAVE THE TRANSFORMER ARTIFACT
            # This is the critical step for your new project. 
            # You must save this to 'artifacts/data_transformation/transformer.joblib'
            transformer_path = os.path.join(self.config.root_dir, "transformer.joblib")
            joblib.dump(preprocessing_obj, transformer_path)
            logger.info(f"Saved preprocessing object to {transformer_path}")
            
            # 5. Reconstruct DataFrame with new numeric column names
            column_names = preprocessing_obj.get_feature_names_out()
            X_df = pd.DataFrame(X_encoded, columns=column_names)

            # Combine the encoded features back with the result column
            final_df = pd.concat([X_df, y.reset_index(drop=True)], axis=1)

            # 6. Split and Save Clean Numeric CSVs
            train, test = train_test_split(final_df, test_size=0.2)
            
            train_path = os.path.join(self.config.root_dir, "train.csv")
            test_path = os.path.join(self.config.root_dir, "test.csv")
            
            train.to_csv(train_path, index=False)
            test.to_csv(test_path, index=False)
            
            logger.info("Split data into numeric train and test sets")
            
            return (train_path, test_path)

        except Exception as e:
            raise e