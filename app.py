import os
import sys
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#from sklearn.datasets import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from autofeat import AutoFeatRegressor
from sklearn.pipeline import make_pipeline
import pickle
'''
def main():
    # Get the dataset from the users GitHub repository
    dataset_path = "https://raw.githubusercontent.com/" + os.environ["GITHUB_REPOSITORY"] +"/master/dataset.csv"
    dataset = pd.read_csv(dataset_path)
    print()
    print(dataset.describe())
    test_autofeat(dataset)
    '''
def load_regression_dataset(datasetpath="https://raw.githubusercontent.com/" + os.environ["GITHUB_REPOSITORY"] +"/master/dataset.csv"):
    dataset = pd.read_csv(dataset_path)
    print()
    print(dataset.describe())
    test_autofeat(dataset)
def test_model(dataset, model, param_grid):
    # load data
    X, y, _ = load_regression_dataset(dataset)
    # split in training and test parts
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)
    if model.__class__.__name__ == "SVR":
        sscaler = StandardScaler()
        X_train = sscaler.fit_transform(X_train)
        X_test = sscaler.transform(X_test)
    # train model on train split incl cross-validation for parameter selection
    gsmodel = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', cv=5)
    gsmodel.fit(X_train, y_train)
    print("best params:", gsmodel.best_params_)
    print("best score:", gsmodel.best_score_)
    print("MSE on training data:", mean_squared_error(y_train, gsmodel.predict(X_train)))
    print("MSE on test data:", mean_squared_error(y_test, gsmodel.predict(X_test)))
    print("R^2 on training data:", r2_score(y_train, gsmodel.predict(X_train)))
    print("R^2 on test data:", r2_score(y_test, gsmodel.predict(X_test)))
    return gsmodel.best_estimator_
def test_autofeat(dataset, feateng_steps=2):
    # load data
    X, y, units = load_regression_dataset(dataset)
    # split in training and test parts
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)
    # run autofeat
    afreg = AutoFeatRegressor(verbose=1, feateng_steps=feateng_steps, units=units)
    # fit autofeat on less data, otherwise ridge reg model with xval will overfit on new features
    X_train_tr = afreg.fit_transform(X_train, y_train)
    X_test_tr = afreg.transform(X_test)
    print("autofeat new features:", len(afreg.new_feat_cols_))
    print("autofeat MSE on training data:", mean_squared_error(y_train, afreg.predict(X_train_tr)))
    print("autofeat MSE on test data:", mean_squared_error(y_test, afreg.predict(X_test_tr)))
    print("autofeat R^2 on training data:", r2_score(y_train, afreg.predict(X_train_tr)))
    print("autofeat R^2 on test data:", r2_score(y_test, afreg.predict(X_test_tr)))
    # train rreg on transformed train split incl cross-validation for parameter selection
    print("# Ridge Regression")
    rreg = Ridge()
    param_grid = {"alpha": [0.00001, 0.0001, 0.001, 0.01, 0.1, 1., 2.5, 5., 10., 25., 50., 100., 250., 500., 1000., 2500., 5000., 10000.]}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gsmodel = GridSearchCV(rreg, param_grid, scoring='neg_mean_squared_error', cv=5)
        gsmodel.fit(X_train_tr, y_train)
    print("best params:", gsmodel.best_params_)
    print("best score:", gsmodel.best_score_)
    print("MSE on training data:", mean_squared_error(y_train, gsmodel.predict(X_train_tr)))
    print("MSE on test data:", mean_squared_error(y_test, gsmodel.predict(X_test_tr)))
    print("R^2 on training data:", r2_score(y_train, gsmodel.predict(X_train_tr)))
    print("R^2 on test data:", r2_score(y_test, gsmodel.predict(X_test_tr)))
    print("# Random Forest")
    rforest = RandomForestRegressor(n_estimators=100, random_state=13)
    param_grid = {"min_samples_leaf": [0.0001, 0.001, 0.01, 0.05, 0.1, 0.2]}
    gsmodel = GridSearchCV(rforest, param_grid, scoring='neg_mean_squared_error', cv=5)
    gsmodel.fit(X_train_tr, y_train)
    print("best params:", gsmodel.best_params_)
    print("best score:", gsmodel.best_score_)
    print("MSE on training data:", mean_squared_error(y_train, gsmodel.predict(X_train_tr)))
    print("MSE on test data:", mean_squared_error(y_test, gsmodel.predict(X_test_tr)))
    print("R^2 on training data:", r2_score(y_train, gsmodel.predict(X_train_tr)))
    print("R^2 on test data:", r2_score(y_test, gsmodel.predict(X_test_tr)))

    if gsmodel:
        pickle.dump(gsmodel,open('model.pkl','wb')) # store the artifact in docker container

        if not os.environ["INPUT_MYINPUT"] == 'zeroinputs':
            inputs = ast.literal_eval(os.environ["INPUT_MYINPUT"])
            print("\nThe Predicted Ouput is :")
            output = gsmodel.predict([inputs])
            print(output)
        else:
            output = ["None"]
            print("\nUser didn't provided inputs to predict")
            
            print("\n=======================Action Completed========================")
            print(f"::set-output name=myOutput::{output[0]}")

        


if __name__ == "_load_regression_dataset_":
    load_regression_dataset()

