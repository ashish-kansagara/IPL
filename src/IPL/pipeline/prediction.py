import joblib
import pandas as pd
from pathlib import Path

class PredictionPipeline:
    def __init__(self):
        # Load both artifacts
        self.model = joblib.load(Path('artifacts/model_trainer/model.joblib'))
        self.transformer = joblib.load(Path('artifacts/data_transformation/transformer.joblib'))

    def predict_proba(self, dataframe):
        # 1. Take the categorical dataframe from the Flask form
        # 2. Use the transformer to turn text into the exact numeric format
        data_numeric = self.transformer.transform(dataframe)
        
        # 3. Pass the numeric data to the model
        return self.model.predict_proba(data_numeric)