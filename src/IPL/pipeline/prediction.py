import joblib
import pandas as pd
from pathlib import Path

class PredictionPipeline:
    def __init__(self):

        self.model = joblib.load(Path('artifacts/model_trainer/model.joblib'))
        self.transformer = joblib.load(Path('artifacts/data_transformation/transformer.joblib'))

    def predict_proba(self, dataframe):
        data_numeric = self.transformer.transform(dataframe)
      
        return self.model.predict_proba(data_numeric)