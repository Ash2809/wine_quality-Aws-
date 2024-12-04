import pandas as pd
import numpy as np
import mlflow

from mlflow.models.signature import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, accuracy_score
from urllib.parse import urlparse

def hyperparameter_tuning(x_train, y_train, params):
    rf = RandomForestClassifier()
    grid_search = GridSearchCV(estimator = rf, cv = 3, n_jobs = -1, param_grid = params)
    grid_search.fit(x_train, y_train)
    return grid_search

def train(input):
    data = pd.read_csv(input)
    y = data['quality']
    x = data.drop(columns=['quality'], axis = 1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 42, test_size = 0.2)

    signature = infer_signature(x_train, y_train)
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    with mlflow.start_run():
        param_grid = {
            'n_estimators' : [100,200,50],
            'max_depth' : [2,5,7]
        }

        grid_search = hyperparameter_tuning(x_train, y_train, param_grid)

        model = grid_search.best_estimator_

        accuracy = accuracy_score(x_test, y_test)

        mlflow.log_param("best n_estimators", grid_search.best_params_['n_estimators'])
        mlflow.log_param("best max_depth", grid_search.best_params_['max_depth'])
        mlflow.log_metric("Accuracy is:", accuracy)

        tracking_url_type_store=urlparse(mlflow.get_tracking_uri()).scheme

        if tracking_url_type_store !='file':
            mlflow.sklearn.log_model(model,"model",registered_model_name="Best Randomforest Model")
        else:
            mlflow.sklearn.log_model(model,"model",signature=signature)

if __name__ == "__main__":
    path = r"C:\MLOPS\wine_quality-Aws-\data\data.csv"
    train(path)


    