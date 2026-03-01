import os
import urllib.request as request
import zipfile
import pandas as pd
from IPL import logger
from IPL.utils.common import get_size
from IPL.entity.config_entity import DataValidationConfig



class DataValidation:
    def __init__(self, config: DataValidationConfig,schema: dict):
        self.config = config
        self.schema = schema
    
    def validate_all_columns(self)-> bool:
        try:
            validation_status = None

            data = pd.read_csv(self.config.unzip_data_dir)
            all_cols = list(data.columns)
            expected_schema = self.schema['COLUMNS']
            all_schema = self.config.all_schema.keys()

            
            for col in all_cols:
                if col not in all_schema:
                    validation_status = False
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f"Validation status: {validation_status}")
                else:
                    validation_status = True
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f"Validation status: {validation_status}")
                        
            for col, expected_type in expected_schema.items():
                
                actual_type = str(data[col].dtype)
                if actual_type != expected_type:
                    validation_status = False
                    logger.error(f"Column '{col}': Expected {expected_type}, but got {actual_type}")
                else:
                    logger.info(f"Column '{col}': Type check passed.")

            return validation_status
        
        except Exception as e:
            raise e