# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/titanic/train.csv")

test = pd.read_csv("/kaggle/input/titanic/test.csv")

res =  pd.read_csv("/kaggle/input/titanic/gender_submission.csv")

test["Ticket"]
data.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1, inplace=True)

test.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1, inplace=True)
data.info()

test.info()
import matplotlib.pyplot as plt

data.hist(bins=50, figsize=(10,8))

plt.show()
from sklearn.model_selection import train_test_split



train_data_set, test_data_set = train_test_split(data, test_size=0.2, random_state=37)



train_corr_matrix = train_data_set.corr()



from pandas.plotting import scatter_matrix

scatter_matrix(train_corr_matrix, figsize=(20,10))
train_data = train_data_set.drop("Survived", axis=1)

train_labels = train_data_set["Survived"].copy()

train_data.info()
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import LabelEncoder



sex = train_data["Sex"].copy()

lb_enc = LabelEncoder()

sex_emc = lb_enc.fit_transform(sex)

sex_enc_pd = pd.DataFrame(sex_emc)





train_data_num = train_data.drop(["Sex", "Embarked"], axis=1)

train_data_num.describe()



sim_imp = SimpleImputer(strategy="median")

processed_num_data = sim_imp.fit_transform(train_data_num)



processed_num_data

train_data_tr = pd.DataFrame(processed_num_data, columns=train_data_num.columns, index=train_data.index)

train_data_tr.info()
train_data["Embarked"].fillna("S", inplace=True)

embarked = train_data["Embarked"].copy()

embarked_encoder = LabelEncoder()

embarked_encoded = embarked_encoder.fit_transform(embarked)

emb_enc_pd = pd.DataFrame(embarked_encoded)

# train_data.drop("Ticket", axis=1, inplace=True)

train_data.info()
num_attribs = list(train_data_num)

num_attribs



cat_attribs = ['Sex', 'Embarked']
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer



num_pipeline = Pipeline([("imputer", SimpleImputer(strategy='median')),

                        ("std_scaler", StandardScaler()),])



full_pipeline = ColumnTransformer([("num", num_pipeline, num_attribs),

                                  ("cat", OneHotEncoder(), cat_attribs)])
processed_full = full_pipeline.fit_transform(train_data)

processed_full
# test_data_modded = test_data_set.drop(["Name", "Cabin", "Survived", "Ticket"], axis=1)
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import cross_val_score



pro_df = pd.DataFrame(processed_full)

print(pro_df.info())



ra_fo = RandomForestClassifier()

ra_fo.fit(processed_full, train_labels)

ra_fo_predictions = ra_fo.predict(processed_full)









cross_val = cross_val_score(ra_fo, processed_full, train_labels,

                           scoring="neg_mean_squared_error", cv=10)

rmse_cross_val = np.sqrt(-cross_val)

rmse_cross_val
from sklearn.svm import SVC

svc = SVC(C=10, gamma=0.1)

svc.fit(processed_full, train_labels)



cross_val_svc = cross_val_score(svc, processed_full, train_labels,

                           scoring="neg_mean_squared_error", cv=10)

rmse_cross_val_svc = np.sqrt(-cross_val_svc)

rmse_cross_val_svc
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score



dec_tree = DecisionTreeClassifier()

dec_tree.fit(processed_full, train_labels)

predic = dec_tree.predict(processed_full)

accuracy = accuracy_score(train_labels, predic)

print(accuracy)



accuracy_ra_fo = accuracy_score(ra_fo_predictions, train_labels)

print(accuracy_ra_fo)
# test_data_set.drop(["Name", "Ticket", "Cabin"], axis=1, inplace=True)

test_data_labels = test_data_set["Survived"].copy()

test_data_set
test_data_set["Embarked"].fillna("S", inplace=True)

test_data_prepared = full_pipeline.fit_transform(test_data_set)

test_data_prepared
test_pred = dec_tree.predict(test_data_prepared)

accuracy_test_dt = accuracy_score(test_data_labels, test_pred)

accuracy_test_dt
ra_test_pred = ra_fo.predict(test_data_prepared)

accuracy_test_ra = accuracy_score(test_data_labels, ra_test_pred)

accuracy_test_ra
svc_test_pred = svc.predict(test_data_prepared)

accuracy_test_svc = accuracy_score(test_data_labels, svc_test_pred)

accuracy_test_svc
score = cross_val_score(svc, processed_full, train_labels,

                        scoring="accuracy", cv=10)



#from sklearn.model_selection import RandomizedSearchCV

#random_grid = {'kernel': ["rbf"],

# 'C': [1, 10, 100, 1000, 10000],

# 'gamma': [0.01,0.1,1]}

'''

svc_random = RandomizedSearchCV(estimator = svc, param_distributions = random_grid, n_iter = 10, cv = 3, verbose=2, random_state=42)

svc_random.fit(processed_full, train_labels)

svc_random.best_params_

'''
{'bootstrap': [True, False],

 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],

 'max_features': ['auto', 'sqrt'],

 'min_samples_leaf': [1, 2, 4],

 'min_samples_split': [2, 5, 10],

 'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}



'''

rf_random = RandomizedSearchCV(estimator = ra_fo, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

rf_random.fit(processed_full, train_labels)

rf_random.best_params_

'''
test["Embarked"].fillna("S", inplace=True)

test_prepared = full_pipeline.fit_transform(test)

test_prepared



test.info()
svc_test_pred_final = svc.predict(test_prepared)

svc_test_pred_final

pred_pd = pd.Series(svc_test_pred_final, name="Survived")

rafo_test_pred_final = ra_fo.predict(test_prepared)

rafo_test_pred_final

pred_pd = pd.Series(rafo_test_pred_final, name="Survived")
results = pd.concat([pd.read_csv('/kaggle/input/titanic/test.csv')['PassengerId'],pred_pd],axis=1)

results.to_csv("predection_mine.csv",index=False)

print(results)

#print(res)
results["Survived"].value_counts()
res["Survived"].value_counts()