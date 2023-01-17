# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# common imports

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import re as re

import seaborn as sns

np.random.seed(28)



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
titanic = pd.read_csv('../input/train.csv')
titanic.head(3)
titanic.info()
titanic["Sex"].value_counts()
titanic["Embarked"].value_counts()
# encode categorical attributes

titanic["Sex_enc"] = titanic["Sex"].map({"female":1, "male": 0}) # encode sex

titanic["Embarked_enc"] = titanic["Embarked"].map({"S":0, "C":1, "Q":2}) # encode Embarked
# Take a look at the distribution of all feature values

%matplotlib inline

titanic.hist(bins=50, figsize=(20,15))

plt.show()
# Take a look at how Embarked, Parch, Pclass, SibSp and Sex are associated with Survived

fig, axes = plt.subplots(2,3, figsize=(16,12))

attr_ls = ["Sex", "Parch", "Pclass", "SibSp", "Embarked"]

for attr, ax in zip(attr_ls[:3], axes[0]):

    titanic.pivot_table("PassengerId", attr, "Survived", "count").plot(kind="bar", stacked=True, ax=ax)

for attr, ax in zip(attr_ls[3:], axes[1]):

    titanic.pivot_table("PassengerId", attr, "Survived", "count").plot(kind="bar", stacked=True, ax=ax) 
# verify the observations with correlation matrix

corr = titanic[["Survived", "Sex_enc", "Parch", "Pclass", "SibSp", "Embarked_enc"]].corr()

np.abs(corr["Survived"]).sort_values(ascending=False)
# what about combining SibSp and Parch

titanic["family_size"] = titanic["Parch"] + titanic["SibSp"]

titanic[["Survived", "family_size"]].corr()["Survived"]
# family size is not more correlated than Parch and SibSp, so delete it

titanic.drop("family_size", axis=1, inplace=True)
# take a look at the numeric features Age, Fare

fig, axes = plt.subplots(1,2,figsize=(12,6))

titanic.boxplot(by="Survived", column="Age", ax=axes[0])

titanic.boxplot(by="Survived", column="Fare", ax=axes[1])

plt.show()
# still take a look at the correlation

titanic[["Survived", "Fare", "Age"]].corr()["Survived"]
# Extract title from Name

def get_title(name):

    title_search = re.search(' ([A-Za-z]+)\.', name)

    if title_search:

        return title_search.group(1)

    return ""

non_rare = ["Mr", "Miss", "Mrs", "Master"] # non rare titles

title = titanic["Name"].map(get_title)

title = np.where(title.isin(non_rare), title, "Rare")

titanic["Title"] = title
titanic["Title"].value_counts()
def extract_cabin(cabin):

    split = re.split("\s", cabin)        

    cabin_letter = [s[0] for s in split]

    if len(set(cabin_letter)) == 1: # if the elements in the list are the same, 

        return cabin_letter[0] # return the element

    else:

        return "M" # otherwise return "M"

titanic["Cabin_letter"] = titanic["Cabin"].map(lambda x: "U" if pd.isnull(x) else extract_cabin(x))
titanic["Cabin_letter"].value_counts()
def ticket_len(ticket):

    split = re.split("\s", ticket)

    return len(split[-1])

titanic["Ticket_len"] = titanic["Ticket"].map(ticket_len)

titanic["Ticket_len"].value_counts()
# lets plot Title, Cabin_letter and Ticket_len

fig, axes = plt.subplots(1,3,  figsize=(16, 6))

attr_ls = ["Title", "Cabin_letter", "Ticket_len"]

for ax, attr in zip(axes, attr_ls):

    titanic.pivot_table("PassengerId", attr, "Survived", "count").plot(kind="bar", stacked=True, colormap="Paired", ax=ax)
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

titanic["Title_enc"] = encoder.fit_transform(titanic["Title"])

titanic["Ticket_enc"] = encoder.fit_transform(titanic["Ticket_len"])

titanic["Cabin_enc"] = encoder.fit_transform(titanic["Cabin_letter"])

corr = titanic.corr()

fig = plt.figure(figsize=(12,10))

sns.heatmap(corr, cmap='BuGn', linewidths=0.1,vmax=1.0, square=True, annot=True)

plt.show()
# defined an estimator to pick attributes

from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelector(BaseEstimator, TransformerMixin):

    def __init__(self, attrib):

        self.attrib = attrib

    def fit(self, X, y=None):

        return self

    def transform(self, X, y=None):

        return X[self.attrib].values
# define a customize binarizer to binarize categorical features (LabelBinarizer() doesn't work with pipeline())

from sklearn.preprocessing import LabelBinarizer

class CustomBinarizer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None,**fit_params):

        return self

    def transform(self, X):

        return LabelBinarizer().fit(X).transform(X)
# label_binarize accepts pre-defined classes

from sklearn.preprocessing import label_binarize

class BinarizerByClass(BaseEstimator, TransformerMixin):

    def __init__(self, classes):

        self.classes = classes

    def fit(self, X, y=None, **fit_params):

        return self

    def transform(self, X):

        return label_binarize(X, classes = self.classes)
class FeatureAdder(BaseEstimator, TransformerMixin):

    def __init__(self, name_attr, cabin_attr):

        self.name_attr = name_attr

        self.cabin_attr = cabin_attr

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        title = X[self.name_attr].map(get_title)

        non_rare = ["Mr", "Miss", "Mrs", "Master"] # non rare titles

        title = np.where(title.isin(non_rare), title, "Rare")

        cabin_letter = X[self.cabin_attr].map(lambda x: "U" if pd.isnull(x) else extract_cabin(x)).values

        return np.c_[X, title, cabin_letter] # append the added columns 
# fill the missing values by mode

class FillingByMode(BaseEstimator, TransformerMixin):

    def __init__(self, attr):

        self.attr = attr

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        X[self.attr].fillna(X[self.attr].mode()[0], inplace=True)

        return X
titanic = pd.read_csv('../input/train.csv')

target = titanic["Survived"]
from sklearn.pipeline import Pipeline

from sklearn.pipeline import FeatureUnion

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import Imputer



featureAdder = FeatureAdder("Name", "Cabin")

titanic = pd.DataFrame(featureAdder.fit_transform(titanic), columns=list(titanic.columns)+["Title", "Cabin_letter"])

titanic.head(3)
cat_pip = FeatureUnion(transformer_list= [

    ("sex_transform", Pipeline([("selector", DataFrameSelector(["Sex"])),

                                ("binarizer", CustomBinarizer())])),  # binarize Sex

    ("embarked_transform", Pipeline([("filling", FillingByMode("Embarked")),

                                     ("selector", DataFrameSelector(["Embarked"])),

                                    ("binarizer", CustomBinarizer())])), # encode Embarked

    ("title_transform", Pipeline([("selector", DataFrameSelector(["Title"])),

                                    ("binarizer", CustomBinarizer())])), # encode Title

    ("cabin_transform", Pipeline([("selector", DataFrameSelector(["Cabin_letter"])),

                                ("binarizer", BinarizerByClass(list(set(titanic["Cabin_letter"]))))])) # encode Cabin Letter

])

cat_pip.fit_transform(titanic)
num_attribs = ["Age", "Fare", "Parch", "SibSp"]

num_pip = Pipeline([

    ("selector", DataFrameSelector(num_attribs)),

    ("imputer", Imputer(strategy="mean")),

    ("std_scaler", StandardScaler())

])
full_pip = FeatureUnion(transformer_list=[

    ("cat_pip", cat_pip),

    ("num_pip", num_pip)

])
prepared_data = full_pip.fit_transform(titanic)

prepared_data = np.c_[titanic["Pclass"].values, prepared_data]

prepared_data.shape