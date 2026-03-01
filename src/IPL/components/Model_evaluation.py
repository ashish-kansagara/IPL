import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, log_loss
from IPL.entity.config_entity import ModelEvaluationConfig
from IPL.utils.common import save_json
from pathlib import Path

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self, actual, pred, pred_proba):
        """
        Metrics suitable for IPL Win Prediction (Classification)
        """
        acc = accuracy_score(actual, pred)
        precision = precision_score(actual, pred)
        recall = recall_score(actual, pred)
        
        loss = log_loss(actual, pred_proba) 
        
        return acc, precision, recall, loss

    def save_results(self):
        
        test_data = pd.read_csv(self.config.test_data_path)
        
        model = joblib.load(self.config.model_path)

        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[self.config.target_column]

        predictions = model.predict(test_x)
       
        pred_proba = model.predict_proba(test_x)

        (acc, precision, recall, loss) = self.eval_metrics(test_y, predictions, pred_proba)
        
        scores = {
            "accuracy": acc, 
            "precision": precision, 
            "recall": recall, 
            "log_loss": loss
        }
        
        save_json(path=Path(self.config.metric_file_name), data=scores)