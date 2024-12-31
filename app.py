import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

import sys
import warnings
import logging
from urllib.parse import urlparse

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file from the URL
    csv_url = (
        "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    )
    try:
        data = pd.read_csv(csv_url, sep=";")
    except Exception as e:
        logger.exception(
            f"Unable to download training & test CSV, check your internet connection. Error: {e}"
        )
        data = None  # Exit safely if data loading fails.

    if data is not None:
        # Split the data into training and test sets. (0.75, 0.25) split.
        train, test = train_test_split(data)

        # The predicted column is "quality" which is a scalar from [3, 9]
        train_x = train.drop(["quality"], axis=1)
        test_x = test.drop(["quality"], axis=1)
        train_y = train["quality"]
        test_y = test["quality"]

        # # Define default values for alpha and l1_ratio
        # in_alpha = 0.5  # Replace with user input or parameter
        # in_l1_ratio = 0.5  # Replace with user input or parameter

        # alpha = 0.5 if in_alpha is None else float(in_alpha)
        # l1_ratio = 0.5 if in_l1_ratio is None else float(in_l1_ratio)


        alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
        l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

        # Useful for multiple runs (only doing one run in this sample notebook)
        with mlflow.start_run():
            # Execute ElasticNet
            lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
            lr.fit(train_x, train_y)

            # Evaluate Metrics
            predicted_qualities = lr.predict(test_x)
            
            (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

            # Print out metrics
            print(f"ElasticNet model (alpha={alpha:f}, l1_ratio={l1_ratio:f}):")
            print(f"  RMSE: {rmse}")
            print(f"  MAE: {mae}")
            print(f"  R2: {r2}")

            # Infer model signature
            predictions = lr.predict(train_x)
            signature = infer_signature(train_x, predictions)

            # Log parameter, metrics, and model to MLflow
            mlflow.log_param("alpha", alpha)
            mlflow.log_param("l1_ratio", l1_ratio)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mae", mae)

            ## for remote server(DAGSHUB)
            remote_server_uri = "https://dagshub.com/Subrat1920/mlflow_tutorial.mlflow"

            mlflow.set_tracking_uri(remote_server_uri)

            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            ## model registry doest not work with file store
            if tracking_url_type_store != 'file':
                ## register the model
                ## There are other ways to use the Model Registry, which depends on the use case,
                ## please refer to the doc for more information:
                ## https://mlflow.org/docs/latest/model-registry.html

                mlflow.sklearn.log_model(lr, 'model', registered_model_name='ElasticNetWineModel', signature=signature)
            else:
                mlflow.sklearn.log_model(lr, "model", signature=signature)
    else:
        logger.error("Data could not be loaded. Exiting script.")
