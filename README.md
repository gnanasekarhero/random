# random

# What does the GitHub action do? :loudspeaker:
It automates your Machine Learning model training, with the help of [autofeat](https://pypi.org/project/autofeat/) python package. This GitHub action makes use of `AutoFeatClassifier()` to build to the classification model. 

Disclaimer: For models with very low samples, they might overfit on noise in the dataset, i.e., find some new features that lead to a good prediction on the training set but result in poor performance on new test samples. 
# How to use? :information_desk_person:
You will find my "Automated ML Models using Autofeat package" action in the [marketplace](https://github.com/marketplace/actions/automated-ml-models-using-autofeat-package). 
Then follow these simple steps:
- Click on **Use latest version**, a prompt will be displayed. We need to copy the code and use it in our workflow as shown below. ![Alt Text](https://dev-to-uploads.s3.amazonaws.com/i/y6hks38kd4jgozaxu1w5.png)
- Make your workflow as shown below. (I have used the Iris dataset as an example)
```
name: Iris Dataset Classifier
on: [push]
jobs:
  build_model:
    runs-on: ubuntu-latest
    steps:
    - name: Train the model
      id: model
      uses: Haimantika/random@master
      with:
        myInput: "[5.1,3.5,1.4,2.1,1.8,0.2]"
    - name: Upload artifact
      uses: actions/upload-artifact@v2
      with:
        name: my-artifact
        path: model.pkl
```
- Create a repository and upload the dataset to train the model. 
  The name of the dataset file must be ***dataset.csv***
- Go to ***Actions*** on GitHub Console and click on ***Set up a workflow yourself***.
- Click on ***Start commit***, give a commit message (optional) and click on ***Commit new file***. The workflow begins soon after this.
- Click on the workflow and you’ll get the console output. You can click on ***Artifacts*** and download the model file keep it for later use.



