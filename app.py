import numpy as np
import pandas as pd
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

    for steps in range(5):
        np.random.seed(55)
        print("### AutoFeat with %i feateng_steps" % steps)
        afreg = AutoFeatRegressor(verbose=1, feateng_steps=steps)
        df = afreg.fit_transform(data, target)
        r2 = afreg.score(data, target)
        print("## Final R^2: %.4f" % r2)
        plt.figure()
        plt.scatter(afreg.predict(data), target, s=2);
        plt.title("%i FE steps (R^2: %.4f; %i new features)" % (steps, r2, len(afreg.new_feat_cols_)))
    afreg = AutoFeatRegressor(verbose=1, feateng_steps=3)
    # train on noisy data
    df = afreg.fit_transform(data, target_noisy)
    # test on real targets
    print("Final R^2: %.4f" % afreg.score(df, target))
    plt.figure()
    plt.scatter(afreg.predict(df), target, s=2);
    afreg = AutoFeatRegressor(verbose=1, feateng_steps=3)
    # train on noisy data
    df = afreg.fit_transform(data, target_very_noisy)
    # test on real targets
    print("Final R^2: %.4f" % afreg.score(df, target))
    plt.figure()
    plt.scatter(afreg.predict(df), target, s=2);

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
