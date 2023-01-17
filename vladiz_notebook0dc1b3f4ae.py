%matplotlib inline

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.




import matplotlib.pyplot as plt

import seaborn as sns
tr_df = pd.read_csv("../input/train.csv")

tr_df.head(3)
tr_df.info()
tr_df.describe()
test_df = pd.read_csv("../input/test.csv")

test_df.head(3)
test_df.info()
test_df.describe()
tr_df[["Age", "Pclass", "Fare", "Parch", "SibSp"]].hist(align='left', figsize = (10,5))

plt.tight_layout()

plt.show()
tr_df.groupby("Sex")["Sex"].count().plot(kind='bar')

plt.show()
tr_df.groupby("Survived")["Survived"].count().plot(kind='bar')

plt.show()
tr_df.groupby("Embarked")["Embarked"].count().plot(kind='bar')

plt.show()
sns.heatmap(tr_df.corr(), annot = True)
def df_transform(df):

    df = df.drop(["Name","Ticket","PassengerId","Cabin","Age"], axis=1)

    df = pd.get_dummies(df, columns=['Sex','Embarked']).fillna(0)

    return df
tr_df = df_transform(tr_df)

tr_df.head(3)
fig, ax = plt.subplots(figsize=(10,10))         # Sample figsize in inches

sns.heatmap(tr_df.corr(), annot = True)
from sklearn.model_selection import train_test_split



#split data to training : test (70% : 30%)

attr_train, attr_test, l_train, l_test = train_test_split(tr_df.drop("Survived", axis=1), tr_df.Survived, train_size = 0.7)
from sklearn.linear_model import LinearRegression



model = LinearRegression()



trained = model.fit(attr_train, l_train)

lin_reg_score = trained.score(attr_test,l_test)

print("LinearRegression score: {}".format(lin_reg_score ))

#output.to_csv("../output/my_gender_submission.csv")
# Time for cross validation with initially selected 70% data for training



from sklearn.model_selection import cross_val_score, StratifiedKFold



attributes = attr_train

labels = l_train



kfold = StratifiedKFold(n_splits = 5)



lin_reg_scores = cross_val_score(model, attributes, labels, cv = kfold)

print(lin_reg_scores)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()



trained = model.fit(attr_train, l_train)

logis_reg_score = trained.score(attr_test,l_test)

print("LogisticRegression score: {}".format(logis_reg_score))
# Loogistic regressioin cross validation



attributes = attr_train

labels = l_train



kfold = StratifiedKFold(n_splits = 5)



logis_reg_scores = cross_val_score(model, attributes, labels, cv = kfold)

print(logis_reg_scores)
from sklearn.model_selection import GridSearchCV



tuned_params = [{"C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],  "penalty": ["l1", "l2"]}]

grid = GridSearchCV(LogisticRegression(C = 1), tuned_params)

grid.fit(attr_train, l_train)

print(grid.best_params_) # Estimator: grid.best_estimator_
# Loogistic regressioin cross validation with tuned parameters



model = LogisticRegression(C = 10, penalty = "l1")



kfold = StratifiedKFold(n_splits = 5)



scores = cross_val_score(model, attributes, labels, cv = kfold)

print(scores)
from sklearn.svm import SVC



model = SVC()



trained = model.fit(attr_train, l_train)

svm_score = trained.score(attr_test,l_test)

print("SVC score: {}".format(svm_score))
# SVC cross validation



attributes = attr_train

labels = l_train



kfold = StratifiedKFold(n_splits = 5)



svm_scores = cross_val_score(model, attributes, labels, cv = kfold)

print(svm_scores)


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()



trained = model.fit(attr_train, l_train)

rand_forest_score = trained.score(attr_test,l_test)

print("RandomForestClassifier score: {}".format(rand_forest_score))
# RandomForestClassifier cross validation



kfold = StratifiedKFold(n_splits = 5)



rand_forest_scores = cross_val_score(model, attributes, labels, cv = kfold)

print(rand_forest_scores)
from sklearn.neighbors import KNeighborsClassifier



model = KNeighborsClassifier()



trained = model.fit(attr_train, l_train)

k_neigh_score = trained.score(attr_test,l_test)

print("KNeighborsClassifier score: {}".format(k_neigh_score))



# KNeighborsClassifier cross validation



kfold = StratifiedKFold(n_splits = 5)



k_neigh_scores = cross_val_score(model, attributes, labels, cv = kfold)

print(k_neigh_scores)
from sklearn.tree import DecisionTreeClassifier



model = DecisionTreeClassifier()



trained = model.fit(attr_train, l_train)

decision_tree_score = trained.score(attr_test,l_test)

print("DecisionTreeClassifier score: {}".format(decision_tree_score))



# KNeighborsClassifier cross validation



kfold = StratifiedKFold(n_splits = 5)



decision_tree_scores = cross_val_score(model, attributes, labels, cv = kfold)

print(decision_tree_scores)
print("Single Score\n\nLinear regressoin:\t {}\nLogistic regression:\t {}\nSVM:\t\t\t {}\nRandom forest:\t\t {}\nK neighbors:\t\t {}\nDecision tree:\t\t {}\n"

      .format(lin_reg_score, logis_reg_score, svm_score, rand_forest_score, k_neigh_score, decision_tree_score))



print("Cross Validation\n\nLinear regressoin:\t {}\nLogistic regression:\t {}\nSVM:\t\t\t {}\nRandom forest:\t\t {}\nK neighbors:\t\t {}\nDecision tree:\t\t {}\n"

      .format(lin_reg_scores, logis_reg_scores, svm_scores, rand_forest_scores, k_neigh_scores, decision_tree_scores))
# Export data



## This is for the submission



#model = LogisticRegression() # may select another model

#trained = model.fit(tr_df.drop("Survived", axis=1), tr_df.Survived)

#predicted = model.predict(df_transform(test_df))

#print(trained)

#print(predicted.round(0).astype(int))



#output = test_df[["PassengerId"]]

#output["Survived"] = predicted.round(0).astype(int)

#output = output.set_index("PassengerId")

#output