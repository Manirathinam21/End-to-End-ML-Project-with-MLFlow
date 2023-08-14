import os
import pandas as pd
from sklearn.model_selection import train_test_split
from mlProject import logger
from mlProject.entity.config_entity import DataTransformationConfig

class DataTransformation:
    def __init__ (self, config: DataTransformationConfig):
        self.config= config

    """Note: you can add different data transformation techniques such as scaler, PCA and all kinds of
     EDA in ML cycle here before passing this data to the model
     
    I'm only adding train_test_splitting becoz this data is already cleaned up"""

    def train_test_splitting(self):
        data= pd.read_csv(self.config.data_path)

        # split the data into training and test sets (0.70 , 0.30) split.
        train, test= train_test_split(data, test_size=0.30)

        train.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)

        logger.info("splited data into train and test sets")
        logger.info(train.shape)
        logger.info(test.shape)

        print(train.shape)
        print(test.shape)