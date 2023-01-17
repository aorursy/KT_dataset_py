import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# for the notebook
# %matplotlib inline
# disable warnings
import warnings
warnings.filterwarnings('ignore')
# get the data and show basic information
train_set = pd.read_csv("../input/train.csv")
train_set.info()
train_set.hist(bins=20, figsize=(16,8))
plt.show()
print("Age.mean =", train_set["Age"].mean())
print("Survived.mean =", train_set["Survived"].mean())
# we make a new copy of the data to work with
t = train_set.copy()
attrs_family_related = ["Family", "Parch", "SibSp"]
t["Family"] = t["Parch"] + t["SibSp"]
t[attrs_family_related].hist(bins=10, figsize=(16,8))
plt.show()
t[attrs_family_related].hist(bins=10, figsize=(16, 8), weights=t["Survived"])
plt.show()
def plot_survival_per_feature(data, feature):
    grouped_by_survival = data[feature].groupby(data["Survived"])
    survival_per_feature = pd.DataFrame({"Survived": grouped_by_survival.get_group(1),
                                        "didnt_Survived": grouped_by_survival.get_group(0),
                                        })
    hist = survival_per_feature.plot.hist(bins=20, alpha=0.6)
    hist.set_xlabel(feature)
    plt.show()
plot_survival_per_feature(t, "Family")
plot_survival_per_feature(t, "Age")
plot_survival_per_feature(t, "Pclass")
t["Embarked"].hist(by=t["Survived"], sharey=True, figsize=(16,8))
plt.show()
t["Embarked"].groupby(t["Pclass"]).value_counts()
# Sex destribution
t["Sex"].value_counts().plot.pie(figsize=(8,8))
plt.show()
# What is the number of survival/non survival in each of the two sex?
t["Sex"].hist(by=t["Survived"], sharey=True, figsize=(16,8))
plt.show()
print(t["Cabin"].dropna())
corr_matrix = t.corr()
corr_matrix["Fare"]
plot_survival_per_feature(t, "Fare")
t["personal_fare"] = t["Fare"] / (t["Family"] + 1)
plot_survival_per_feature(t, "personal_fare")
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
sex = t["Sex"]
sex_encoded = encoder.fit_transform(sex)
t2 = t.copy()
t2["Sex"] = sex_encoded
embarked = t["Embarked"].fillna("C")
embarked_encoded = encoder.fit_transform(embarked)
t2["Embarked"] = embarked_encoded
t2.corr()["Embarked"]
labels = train_set["Survived"]
features_data = train_set.drop("Survived", axis=1)
from sklearn.preprocessing import Imputer, LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
# store the columns for the learning_data
COLUMNS = None

class Dropper(BaseEstimator, TransformerMixin):
    def __init__(self, to_drop=["PassengerId", "Name", "Ticket", "Cabin"]):
        self.to_drop = to_drop
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return X.drop(self.to_drop, axis=1)
    
    
class AttributesExtension(BaseEstimator, TransformerMixin):
    def __init__(self, family=True, personal_fare=True, is_child=True, is_child_and_sex=True):
        self.family = family
        self.personal_fare = personal_fare
        self.is_child = is_child
        self.is_child_and_sex = is_child_and_sex
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        if self.family:
            family = X["Parch"] + X["SibSp"]
            X["Family"] = family
        if self.personal_fare and self.family:
            personal_fare = X["Fare"] / (X["Family"] + 1)
            X["Personal_fare"] = personal_fare
        # is_child improved the model by 2% accuracy
        if self.is_child:
            X["is_child"] = X["Age"] <= 8
        if self.is_child_and_sex:
            X["is_child_and_sex"] = X["Sex"] * X["is_child"]
        
            
        #save columns
        global COLUMNS
        COLUMNS = X.columns.tolist()
        return X
    
    
class AttributesEncoding(BaseEstimator, TransformerMixin):
    def __init__(self, sex=True, embarked=True):
        self.sex = sex
        self.embarked = embarked
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        encoder = LabelEncoder()
        if self.sex:
            sex_encoded = encoder.fit_transform(X["Sex"])
            X["Sex"] = sex_encoded
        if self.embarked:
            #impute with C
            embarked_encoded = encoder.fit_transform(X["Embarked"].fillna('C'))
            X["Embarked"] = embarked_encoded
        return X
from sklearn.impute import SimpleImputer
pipeline = Pipeline([
    ('dropper', Dropper()),
    ('encoder', AttributesEncoding()),
    ('extender', AttributesExtension()),
    ('imputer', SimpleImputer(strategy="mean")),
])
learning_data = pipeline.fit_transform(features_data)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm.classes import SVC
from sklearn.metrics import accuracy_score

svc = SVC()
log_reg = LogisticRegression()
#log_reg.fit(learning_data, labels)
rand_for = RandomForestClassifier()
#rand_for.fit(learning_data, labels)

models = {
    "Logistic Regression": log_reg,
    "Random Forest": rand_for,
    "SVM": svc,
}

for model in models.keys():
    scores = cross_val_score(models[model], learning_data, labels, scoring="accuracy", cv=10)
    print("===", model, "===")
    print("scores = ", scores)
    print("mean = ", scores.mean())
    print("variance = ", scores.var())
    models[model].fit(learning_data, labels)
    print("score on the learning data = ", accuracy_score(models[model].predict(learning_data), labels))
    print("")
import eli5
from eli5.sklearn import PermutationImportance

log_reg.fit(learning_data, labels)
perm_imp = PermutationImportance(log_reg, random_state=1).fit(learning_data, labels)
eli5.show_weights(perm_imp,feature_names=COLUMNS)
test_set = pd.read_csv("../input/test.csv")
pred = pipeline.fit_transform(test_set)
log_reg.fit(learning_data, labels)
sub = pd.DataFrame(test_set["PassengerId"], columns=("PassengerId", "Survived"))
sub["Survived"] = log_reg.predict(pred)
#write predicted data to submit it
#sub.to_csv("../input/sub.csv", index=False)