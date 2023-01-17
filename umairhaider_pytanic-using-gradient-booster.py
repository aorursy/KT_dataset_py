import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



train_frame = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )

test_frame  = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )
# Droping columns which are not relevant to train the model

train_frame = train_frame.drop(['Name','Ticket', 'Cabin'], axis=1)

test_frame = test_frame.drop(['Name','Ticket', 'Cabin'], axis=1)
# Method for data cleaning

def fill_data(titanic):

    

    titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median()) # Median will be filled for missing ages

    titanic["Embarked"] = titanic["Embarked"].fillna("S") # We have selected this on the basis of mode

    titanic["Fare"] = titanic["Fare"].fillna(titanic["Fare"].median()) # Median will be filled for missing fares



    return titanic
# Checking the missing values in training dataset before data cleaning

train_frame.isnull().sum()
# Checking the missing values in training dataset after data cleaning

notnull_train = fill_data(train_frame)

notnull_train.isnull().sum()
# Checking the missing values in test dataset after data cleaning

test_frame.isnull().sum()
# Checking the missing values in test dataset after data cleaning

notnull_test = fill_data(test_frame)

notnull_test.isnull().sum()
from sklearn import preprocessing



# Method to convert some columns to numerical data for better computation

def encode_features(frame):

    features = ['Sex', 'Embarked']

    

    for feature in features:

        le = preprocessing.LabelEncoder()

        le = le.fit(frame[feature])

        frame[feature] = le.transform(frame[feature])

    return frame
# Converting some columns of training dataset to numerical values

train_frame_cleaned = encode_features(notnull_train)

train_frame_cleaned
# Converting some columns of test dataset to numerical values

test_frame_cleaned = encode_features(notnull_test)

test_frame_cleaned
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
from sklearn.svm import SVC

from sklearn.model_selection import cross_val_score





#SVC classifier

svc_classifier = SVC(probability=True)





# Calculating score of our model

print("SVC Score:")

cross_val_score(svc_classifier, train_frame_cleaned[features], train_frame_cleaned['Survived']).mean()
from sklearn.ensemble import GradientBoostingClassifier



gbc_classifier = GradientBoostingClassifier()



# Calculating score of our model

print("GBC Score:")

cross_val_score(gbc_classifier, train_frame_cleaned[features], train_frame_cleaned['Survived']).mean()
# Training the model



gbc_classifier.fit(train_frame_cleaned[features], train_frame_cleaned['Survived'])



predictions = gbc_classifier.predict(test_frame_cleaned[features])



output = pd.DataFrame({

        "PassengerId": test_frame_cleaned["PassengerId"],

        "Survived": predictions

    })
# Exporting the result to CSV file

output.to_csv("../working/titanic_submission.csv", index=False)