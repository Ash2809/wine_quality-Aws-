import pandas as pd
import numpy as np
import mlflow
from mlflow.models.signature import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

def train(input):
    data = pd.read_csv(input)
    
    x_train, x_test, y_train, y_test = train_test_split(x)
    