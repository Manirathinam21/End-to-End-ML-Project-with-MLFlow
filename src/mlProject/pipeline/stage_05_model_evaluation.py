from mlProject import logger
from mlProject.config.configuration import ConfigurationManager
from mlProject.components.model_evaluation import ModelEvaluation


STAGE_NAME= 'Model Evaluation Stage'

class ModelEvaluationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config= ConfigurationManager()
        model_evaluation_config= config.get_model_evaluation_config()
        model_evaluation= ModelEvaluation(config= model_evaluation_config)
        model_evaluation.log_into_mlflow()


if __name__ == "__main__":
    try:
        logger.info(f'>>>>>> Stage {STAGE_NAME} started <<<<<<')
        obj=ModelEvaluationTrainingPipeline()
        obj.main()
        logger.info(f'>>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x')
    except Exception as e:
        logger.exception(e)
        raise e