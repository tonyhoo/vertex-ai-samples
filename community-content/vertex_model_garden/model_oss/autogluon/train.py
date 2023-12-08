import os
import argparse
import pandas as pd
from autogluon.tabular import TabularPredictor
import json


class BaseConfig:
    def to_dict(self):
        return {key: value for key, value in self.__dict__.items() if value is not None}


class DataConfig(BaseConfig):
    def __init__(self, train_data_path):
        self.train_data_path = train_data_path


class ProblemConfig(BaseConfig):
    def __init__(self, label, problem_type):
        self.label = label
        self.problem_type = problem_type


class EvaluationConfig(BaseConfig):
    def __init__(self, eval_metric):
        self.eval_metric = eval_metric


class TrainingConfig(BaseConfig):
    def __init__(self, time_limit, presets, hyperparameters):
        self.time_limit = time_limit
        self.hyperparameters = hyperparameters
        self.presets = presets


def parse_args():
    parser = argparse.ArgumentParser(description="AutoGluon Tabular Predictor")
    # Add arguments for each config class
    parser.add_argument("--train_data_path", type=str, required=True, help="Path to the input data CSV file.")
    parser.add_argument("--label", type=str, required=True, help="Target variable column name.")
    parser.add_argument("--problem_type", type=str, choices=["binary", "multiclass", "regression", "quantile"],
                        default=None, help="Problem type.")
    parser.add_argument("--eval_metric", type=str, default=None, help="Evaluation metric to use.")
    # Add arguments for TrainingConfig if needed
    parser.add_argument("--time_limit", type=int, default=None, help="Time limit in seconds for training.")
    parser.add_argument("--presets", type=str, default="medium_quality", help="Presets used for training ")
    parser.add_argument("--hyperparameters", type=json.loads, default=None,
                        help="Hyperparameter dictionary in JSON format.")
    args = parser.parse_args()

    data_config = DataConfig(train_data_path=args.train_data_path)
    problem_config = ProblemConfig(label=args.label, problem_type=args.problem_type)
    eval_config = EvaluationConfig(eval_metric=args.eval_metric)
    training_config = TrainingConfig(time_limit=args.time_limit, presets=args.presets,
                                     hyperparameters=args.hyperparameters)

    return data_config, problem_config, eval_config, training_config


def main():
    """
    To be run in the docker container with sample usage: docker run {image_name} --train_data_path=https://raw.githubusercontent.com/mli/ag-docs/main/knot_theory/train.csv --label=signature
    Returns:

    """
    data_config, problem_config, eval_config, training_config = parse_args()

    # Load the training data
    data = pd.read_csv(data_config.train_data_path)

    # Create a TabularPredictor
    predictor = TabularPredictor(**problem_config.to_dict(), **eval_config.to_dict())

    # Fit the model
    predictor.fit(data, **training_config.to_dict())


if __name__ == "__main__":
    main()
