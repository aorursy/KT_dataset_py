import numpy as np
import pandas as pd
import sklearn as sk
import sklearn.ensemble as en


def clean_data_set(dataset):
    dataset["Age"] = dataset["Age"].fillna(dataset["Age"].median())
    dataset.loc[dataset["Sex"] == "male", "Sex"] = 0
    dataset.loc[dataset["Sex"] == "female", "Sex"] = 1

#Print you can execute arbitrary python code
train = clean_data_set(pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, ))
test = clean_data_set(pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, ))
predictors = ["Pclass", "Sex", "Age", "Fare"]

print(train.describe())


#Print to standard output, and see the results in the "log" section below after running your script
alg = en.RandomForestClassifier(n_estimators=100, min_samples_split=4, min_samples_leaf=4, n_jobs=-1, random_state=1)
alg.fit(train[predictors], train["Survived"])

predictions = alg.predict(test[predictors])
predictions[predictions < .5] = 0
predictions[predictions >= .5] = 1

output = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": predictions.astype(int)})


#Any files you save will be available in the output tab below
output.to_csv('kaggle.csv', index=False)