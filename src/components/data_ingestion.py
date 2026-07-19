import os
import sys
from pathlib import Path
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    train_data_path: str
    test_data_path: str
    raw_data_path: str


class DataIngestion:
    def __init__(self):
        project_root = Path(__file__).resolve().parents[2]
        artifacts_dir = project_root / 'artifacts'
        self.ingestion_config = DataIngestionConfig(
            train_data_path=str(artifacts_dir / 'train.csv'),
            test_data_path=str(artifacts_dir / 'test.csv'),
            raw_data_path=str(artifacts_dir / 'data.csv')
        )
        self.dataset_path = project_root / 'data' / 'stud.csv'

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")

        try:
            df = pd.read_csv(self.dataset_path)
            logging.info(f'Read the dataset as dataframe from {self.dataset_path}')

            os.makedirs(
                os.path.dirname(self.ingestion_config.train_data_path),
                exist_ok=True
            )

            df.to_csv(
                self.ingestion_config.raw_data_path,
                index=False,
                header=True
            )

            logging.info("Train test split initiated")

            train_set, test_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42
            )

            train_set.to_csv(
                self.ingestion_config.train_data_path,
                index=False,
                header=True
            )

            test_set.to_csv(
                self.ingestion_config.test_data_path,
                index=False,
                header=True
            )

            logging.info("Ingestion of data completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()