from IPL.config.configuration import ConfigurationManager
from IPL.components.Model_evaluation import ModelEvaluation
from IPL import logger
from IPL.entity.config_entity import ModelEvaluationConfig

STAGE_NAME="model evaluation stage"

class ModelEvaluationTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation_config = ModelEvaluation(config=model_evaluation_config)
        model_evaluation_config.save_results()
        
if __name__== "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        model_evl = ModelEvaluationTrainingPipeline()
        model_evl.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e 