import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib



import matplotlib.pyplot as plt

from scipy.stats import skew

from scipy.stats.stats import pearsonr





%config InlineBackend.figure_format = 'retina' #set 'png' here when working on notebook

%matplotlib inline





train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



train = train.drop(['Name'], axis=1)

test = test.drop(['Name'], axis=1)



train.head()

#print('_'*80)

#test.head()

all_data = pd.concat((train.loc[:,'Pclass':'Embarked'],

                      test.loc[:,'Pclass':'Embarked']))



#all_data.head()
matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)

prices = pd.DataFrame({"Fare":train["Fare"], "log(Fare + 1)":np.log1p(train["Fare"])})

prices.hist()
#log transform the target:

#train["SalePrice"] = np.log1p(train["SalePrice"])



#log transform skewed numeric features:

numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index



skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness

skewed_feats = skewed_feats[skewed_feats > 0.75]

skewed_feats = skewed_feats.index



all_data[skewed_feats] = np.log1p(all_data[skewed_feats])
all_data = pd.get_dummies(all_data)
#filling NA's with the mean of the column:

all_data = all_data.fillna(all_data.mean())
#print(all_data)
#creating matrices for sklearn:

X_train = all_data[:train.shape[0]]

X_test = all_data[train.shape[0]:]

y = train.Survived

#print(y)
# machine learning

from sklearn.linear_model import LogisticRegression
# Logistic Regression



logreg = LogisticRegression()

logreg.fit(X_train, y)

accuracy = round(logreg.score(X_train, y) * 100, 2)

print(accuracy)



logreg_preds = logreg.predict(X_test)



#Logistic regression submission

#solution = pd.DataFrame({"PassengerId":test.PassengerId, "Survived":logreg_preds})

#solution.to_csv("logreg_solution.csv", index = False)
# Random Forest

from sklearn.ensemble import RandomForestClassifier



random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, y)

random_forest_preds = random_forest.predict(X_test)

random_forest.score(X_train, y)

accuracy = round(random_forest.score(X_train, y) * 100, 2)

print(accuracy)
#Random forest submission

solution = pd.DataFrame({"PassengerId":test.PassengerId, "Survived":random_forest_preds})

solution.to_csv("random_forest_solution.csv", index = False)