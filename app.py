import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from autofeat import FeatureSelector, AutoFeatRegressor
from sklearn.pipeline import make_pipeline
import pickle
def main():
    # Get the dataset from the users GitHub repository
    dataset_path = "https://raw.githubusercontent.com/" + os.environ["GITHUB_REPOSITORY"] +"/master/dataset.csv"
    data = pd.read_csv(dataset_path)
    print()
    print(data.describe())

def test_model(dataset, model, param_grid):
    # load data
    X, y, _ = load_classification_dataset(dataset)
    # split in training and test parts
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)
    if model.__class__.__name__ == "SVC":
        sscaler = StandardScaler()
        X_train = sscaler.fit_transform(X_train)
        X_test = sscaler.transform(X_test)
    # train model on train split incl cross-validation for parameter selection
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gsmodel = GridSearchCV(model, param_grid, cv=5)
        gsmodel.fit(X_train, y_train)
    print("best params:", gsmodel.best_params_)
    print("best score:", gsmodel.best_score_)
    print("Acc. on training data:", accuracy_score(y_train, gsmodel.predict(X_train)))
    print("Acc. on test data:", accuracy_score(y_test, gsmodel.predict(X_test)))
    return gsmodel.best_estimator_
def test_autofeat(dataset, feateng_steps=2):
    # load data
    X, y, units = load_classification_dataset(dataset)
    # split in training and test parts
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)
    # run autofeat
    afreg = AutoFeatClassifier(verbose=1, feateng_steps=feateng_steps, units=units)
    # fit autofeat on less data, otherwise ridge reg model with xval will overfit on new features
    X_train_tr = afreg.fit_transform(X_train, y_train)
    X_test_tr = afreg.transform(X_test)
    print("autofeat new features:", len(afreg.new_feat_cols_))
    print("autofeat Acc. on training data:", accuracy_score(y_train, afreg.predict(X_train_tr)))
    print("autofeat Acc. on test data:", accuracy_score(y_test, afreg.predict(X_test_tr)))
    # train rreg on transformed train split incl cross-validation for parameter selection
    print("# Logistic Regression")
    rreg = LogisticRegression(class_weight="balanced")
    param_grid = {"C": np.logspace(-4, 4, 10)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gsmodel = GridSearchCV(rreg, param_grid, cv=5)
        gsmodel.fit(X_train_tr, y_train)
    print("best params:", gsmodel.best_params_)
    print("best score:", gsmodel.best_score_)
    print("Acc. on training data:", accuracy_score(y_train, gsmodel.predict(X_train_tr)))
    print("Acc. on test data:", accuracy_score(y_test, gsmodel.predict(X_test_tr)))
    print("# Random Forest")
    rforest = RandomForestClassifier(n_estimators=100, random_state=13)
    param_grid = {"min_samples_leaf": [0.0001, 0.001, 0.01, 0.05, 0.1, 0.2]}
    gsmodel = GridSearchCV(rforest, param_grid, cv=5)
    gsmodel.fit(X_train_tr, y_train)
    print("best params:", gsmodel.best_params_)
    print("best score:", gsmodel.best_score_)
    print("Acc. on training data:", accuracy_score(y_train, gsmodel.predict(X_train_tr)))
    print("Acc. on test data:", accuracy_score(y_test, gsmodel.predict(X_test_tr)))

    if pipe:
        pickle.dump(pipe,open('model.pkl','wb')) # store the artifact in docker container

        if not os.environ["INPUT_MYINPUT"] == 'zeroinputs':
            inputs = ast.literal_eval(os.environ["INPUT_MYINPUT"])
            print("\nThe Predicted Ouput is :")
            output = pipe.predict([inputs])
            print(output)
        else:
            output = ["None"]
            print("\nUser didn't provided inputs to predict")
        
        print("\n=======================Action Completed========================")
        print(f"::set-output name=myOutput::{output[0]}")

        


if __name__ == "__main__":
    main()
