import numpy as np
import pandas
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import LabelEncoder
np.random.seed(31)
data = pandas.read_csv('../input/titanic/train.csv')

#It's going to be easier dealing with the data if the target is at the last column of the dataframe
survived = data["Survived"]
data.drop(columns=["Survived"], inplace=True)
data["Survived"] = survived
del survived

#I'm not a big fan of names that aren't explicit, neither should you. 
data = data.rename(columns={"SibSp" : "NumSiblingSpouse", "Parch" : "NumParentChildren"})
data["Embarked"].replace({"S": "Southampton", "C" : "Cherbourg", "Q" : "Queenstown"}, inplace=True)
data_observed = data.copy()
data_observed["Embarked"].fillna("Unavailable", inplace=True)
data_observed["EmbarkedEncoded"] = LabelEncoder().fit_transform(data_observed["Embarked"].values)
data_observed["SexEncoded"] = LabelEncoder().fit_transform(data_observed["Sex"].values)
data_observed.head(5)
print("Survived values :\n" + str(data_observed["Survived"].value_counts()) + "\n---------------------------")
print("NumParentChildren values :\n" + str(data_observed["NumParentChildren"].value_counts()) + "\n---------------------------")
print("Embarked values :\n" + str(data_observed["Embarked"].value_counts()) + "\n---------------------------")
print("NumSiblingSpouse values :\n" + str(data_observed["NumSiblingSpouse"].value_counts()) + "\n---------------------------")
print("Pclass values :\n" + str(data_observed["Pclass"].value_counts()) + "\n---------------------------")
print("Sex values :\n" + str(data_observed["SexEncoded"].value_counts()) + "\n---------------------------")
plt.plot(range(0,len(data_observed)), data_observed["Fare"].sort_values())
plt.title("Sorted 'Fare' feature values")
plt.ylabel("Price")
plt.show()

plt.plot(range(0,len(data_observed)), data_observed["Age"].sort_values())
plt.title("Sorted 'Age' feature values")
plt.ylabel("Age")
plt.show()
data_observed.info()
data_observed.corr()
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree

data_copy = data_observed.copy()
survived = data_copy["Survived"]
data_copy.drop(columns=["PassengerId", "Survived", "Name", "Sex", "Ticket", "Cabin", "Embarked"], inplace=True)
data_copy["Age"].fillna(data_copy["Age"].mean(), inplace=True)
X_train, X_val, y_train, y_val = train_test_split(data_copy, survived, test_size=0.4, random_state=31)

dt = DecisionTreeClassifier(max_depth=4, criterion="entropy", random_state=31)
dt.fit(X_train, y_train)
print("Accuracy score : " + str(dt.score(X_val, y_val)))
data_observed["Title"] = data_observed["Name"].str.extract(r'\,\s(.*)?\w*\.').replace({'Ms' : 'Miss'})
data_observed["TitleEncoded"] = LabelEncoder().fit_transform(data_observed["Title"])
import pandas as pd
from IPython.core.display import HTML

display(HTML(data_observed[["Ticket", "Survived"]].sort_values(by="Survived", ascending=False).to_html()))
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder
import re

extract_title = lambda x : x.str.extract(r'\,\s(.*)?\w*\.').replace({'Ms' : 'Miss'})

data_preparation_pipeline = make_column_transformer(
    (Pipeline([
        ("transform_name", FunctionTransformer(extract_title, validate=False)), 
        ("encode_title", OrdinalEncoder())]), "Name"),
    (SimpleImputer(strategy="mean"), ["Age"]),
    (Pipeline([
        ("impute_embarked", SimpleImputer(strategy="constant", fill_value="Unavailable")), 
        ("encode_embarked", OrdinalEncoder())]), ["Embarked"]),
    (OrdinalEncoder(), ["Sex"]),
    ("drop", ["PassengerId", "Ticket", "Cabin"])
, remainder="passthrough")
survived = data["Survived"]
training_pipeline = Pipeline([("data_preparation",data_preparation_pipeline), ("drop_survived",make_column_transformer(("drop", [-1]), remainder="passthrough"))])

prepared_x = training_pipeline.fit_transform(data)

X_train, X_val, y_train, y_val = train_test_split(prepared_x, survived, test_size=0.4, random_state=31)
accuracy_by_depth = [0,0]
f1_by_depth = [0,0]
for i in range(2, len(data.columns)):
    dt = DecisionTreeClassifier(max_depth=i, min_samples_split=0.01, criterion="entropy", random_state=31)
    
    dt.fit(X_train, y_train)
    accuracy_by_depth.append(dt.score(X_val, y_val))
    f1_by_depth.append(f1_score(y_val, dt.predict(X_val)))
    
plt.plot(range(len(accuracy_by_depth)), accuracy_by_depth)
plt.plot(range(len(accuracy_by_depth)), f1_by_depth)
plt.xlim(2, len(accuracy_by_depth))
plt.legend(["Accuracy", "F1 Score"])
plt.ylabel("Score")
plt.xlabel("Tree depth")
plt.ylim(0.6, 1)
plt.title("Scores by DecisionTree depth")
plt.show()
from matplotlib.pyplot import figure
figure(num=None, figsize=(8, 6), dpi=150, facecolor='w', edgecolor='k')

feature_names = ["Title", "Age", "Embarked", "Sex", "PClass", "# sibling/spouse", "# children/parent", "Fare"]

decision_tree = DecisionTreeClassifier(max_depth=3, criterion="entropy", random_state=31)
decision_tree.fit(X_train, y_train)
plt.show(plot_tree(decision_tree, feature_names=feature_names, class_names=["Died", "Survived"]))

print(dict(zip(feature_names, decision_tree.feature_importances_)))
print("Accuracy : " + str(decision_tree.score(X_val, y_val)))
print("F1 Score : " + str(f1_score(y_val, decision_tree.predict(X_val))))
import joblib

def extract_title(x):
    return x.str.extract(r'\,\s(.*)?\w*\.').replace("Ms", "Miss")

def map_title(x):
    x[~x.isin(["Mr", "Miss", "Mrs"])] = "Other"
    return x

data_preparation_pipeline = make_column_transformer(
    (Pipeline([
        ("transform_name",  FunctionTransformer(extract_title, validate=False)), 
        ("map_name",  FunctionTransformer(map_title, validate=False)),
        ("encode_title", OrdinalEncoder())]), "Name"),
    (SimpleImputer(strategy="most_frequent"), ["Fare"]),
    (SimpleImputer(strategy="most_frequent"), ["NumSiblingSpouse"]),
    (SimpleImputer(strategy="most_frequent"), ["Pclass"]),
    (OrdinalEncoder(), ["Sex"]))

survived = data["Survived"]

#drop the Survived column
data_without_survived = data.drop(columns=["Survived"])

prediction_pipeline = Pipeline([("preparation", data_preparation_pipeline), ("prediction", DecisionTreeClassifier(max_depth=3, criterion="entropy", random_state=31))])
prediction_pipeline.fit(data_without_survived, survived)

joblib.dump(prediction_pipeline, "prediction_pipeline.joblib")
test_data = pandas.read_csv('../input/titanic/test.csv')
test_data = test_data.rename(columns={"SibSp" : "NumSiblingSpouse", "Parch" : "NumParentChildren"})

passengerIds = test_data["PassengerId"]
predictions = prediction_pipeline.predict(test_data)

my_submission = pd.DataFrame({'PassengerId': passengerIds, 'Survived': predictions})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)
pandas.DataFrame(data_preparation_pipeline.fit_transform(test_data), columns=["Title","Sex","Pclass","Fare", "NumSiblingSpouse"]).info()
