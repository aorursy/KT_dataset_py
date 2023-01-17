# Basic libraries

import numpy as np

import pandas as pd

import warnings

warnings.simplefilter('ignore')



# Directry check

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Visualization

from matplotlib import pyplot as plt

import seaborn as sns

sns.set()



# Data preprocessing

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler



# Random forest

from sklearn.ensemble import RandomForestClassifier



# parameter opimization

from sklearn.model_selection import GridSearchCV



# Validation

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
df = pd.read_csv("/kaggle/input/income-classification/income_evaluation.csv", header=0)
df.head()
df.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',

              'marital_status', 'occupation', 'relationship', 'race', 'sex',

              'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']
df["income"].value_counts()
# data size

df.shape
# data info

df.info()
# Null value

df.isnull().sum()
# Unique values

for i in range(15):

    print("-"*50)

    print(df.columns[i])

    print(df.iloc[:,i].value_counts())
# basic statistics balues

# whole data

df.describe()
# income : <=50K

df.query("income==' <=50K'").describe()
# income : >50K

df.query("income==' >50K'").describe()
# Distribution

fig, ax = plt.subplots(3,2, figsize=(25,20), facecolor="lavender")

plt.subplots_adjust(wspace=0.2, hspace=0.3)



# age

sns.distplot(df.query("income==' <=50K'")["age"], ax=ax[0,0], label="<=50K", bins=30, kde=False)

sns.distplot(df.query("income==' >50K'")["age"], ax=ax[0,0], label=">50K", bins=30, kde=False)

ax[0,0].set_title("age distribution")

ax[0,0].set_ylabel("count")

ax[0,0].legend(facecolor="white")



# fnlwgt

sns.distplot(df.query("income==' <=50K'")["fnlwgt"], ax=ax[0,1], label="<=50K", bins=30, kde=False)

sns.distplot(df.query("income==' >50K'")["fnlwgt"], ax=ax[0,1], label=">50K", bins=30, kde=False)

ax[0,1].set_title("fnlwgt distribution")

ax[0,1].set_ylabel("count")

ax[0,1].legend(facecolor="white")



# capital-gain

sns.distplot(np.log(df.query("income==' <=50K'")["capital_gain"]+1), ax=ax[1,0], label="<=50K", bins=30, kde=False)

sns.distplot(np.log(df.query("income==' >50K'")["capital_gain"]+1), ax=ax[1,0], label=">50K", bins=30, kde=False)

ax[1,0].set_title("capital_gain distribution")

ax[1,0].set_xlabel("log(capital_gain)")

ax[1,0].set_ylabel("count")

ax[1,0].set_yscale("log")

ax[1,0].legend(facecolor="white")



# capital-loss

sns.distplot(np.log(df.query("income==' <=50K'")["capital_loss"]+1), ax=ax[1,1], label="<=50K", bins=30, kde=False)

sns.distplot(np.log(df.query("income==' >50K'")["capital_loss"]+1), ax=ax[1,1], label=">50K", bins=30, kde=False)

ax[1,1].set_title("capital_loss distribution")

ax[1,1].set_xlabel("log(capital_loss)")

ax[1,1].set_ylabel("count")

ax[1,1].set_yscale("log")

ax[1,1].legend(facecolor="white")



# education-num

sns.distplot(df.query("income==' <=50K'")["education_num"], ax=ax[2,0], label="<=50K", bins=30, kde=False)

sns.distplot(df.query("income==' >50K'")["education_num"], ax=ax[2,0], label=">50K", bins=30, kde=False)

ax[2,0].set_title("education-num distribution")

ax[2,0].set_ylabel("count")

ax[2,0].legend(facecolor="white")



# hours-per-week

sns.distplot(df.query("income==' <=50K'")["hours_per_week"], ax=ax[2,1], label="<=50K", bins=30, kde=False)

sns.distplot(df.query("income==' >50K'")["hours_per_week"], ax=ax[2,1], label=">50K", bins=30, kde=False)

ax[2,1].set_title("hours-per-week distribution")

ax[2,1].set_ylabel("count")

ax[2,1].legend(facecolor="white")
# scatter plot

g = sns.PairGrid(df, vars=["age", "fnlwgt", "capital_gain","capital_loss", "education_num", "hours_per_week"], hue="income")

g.map(plt.scatter, alpha=0.8)

g.add_legend()
# histgram

fig, ax = plt.subplots(4,2, figsize=(25,30), facecolor="lavender")

plt.subplots_adjust(wspace=0.2, hspace=0.6)



# workclass

sns.countplot("workclass", data=df, hue="income", ax=ax[0,0])

ax[0,0].tick_params(axis='x', labelrotation=90)

ax[0,0].legend(loc='upper right', facecolor='white')

ax[0,0].set_xlabel('')

ax[0,0].set_title("work class")



# education

sns.countplot("education", data=df, hue="income", ax=ax[0,1])

ax[0,1].tick_params(axis='x', labelrotation=90)

ax[0,1].legend(loc='upper right', facecolor='white')

ax[0,1].set_xlabel('')

ax[0,1].set_title("education")



# marital-status

sns.countplot("marital_status", data=df, hue="income", ax=ax[1,0])

ax[1,0].tick_params(axis='x', labelrotation=90)

ax[1,0].legend(loc='upper right', facecolor='white')

ax[1,0].set_xlabel('')

ax[1,0].set_title("marital-status")



# occupation

sns.countplot("occupation", data=df, hue="income", ax=ax[1,1])

ax[1,1].tick_params(axis='x', labelrotation=90)

ax[1,1].legend(loc='upper right', facecolor='white')

ax[1,1].set_xlabel('')

ax[1,1].set_title("occupation")



# relationship

sns.countplot("relationship", data=df, hue="income", ax=ax[2,0])

ax[2,0].tick_params(axis='x', labelrotation=90)

ax[2,0].legend(loc='upper right', facecolor='white')

ax[2,0].set_xlabel('')

ax[2,0].set_title("relationship")



# race

sns.countplot("race", data=df, hue="income", ax=ax[2,1])

ax[2,1].tick_params(axis='x', labelrotation=90)

ax[2,1].legend(loc='upper right', facecolor='white')

ax[2,1].set_xlabel('')

ax[2,1].set_title("race")



# sex

sns.countplot("occupation", data=df, hue="income", ax=ax[3,0])

ax[3,0].tick_params(axis='x', labelrotation=90)

ax[3,0].legend(loc='upper right', facecolor='white')

ax[3,0].set_xlabel('')

ax[3,0].set_title("occupation")



# native-country

sns.countplot("native_country", data=df, hue="income", ax=ax[3,1])

ax[3,1].tick_params(axis='x', labelrotation=90)

ax[3,1].legend(loc='upper right', facecolor='white')

ax[3,1].set_xlabel('')

ax[3,1].set_title("native-country")
# histgram

fig, ax = plt.subplots(4,2, figsize=(25,30), facecolor="lavender")

plt.subplots_adjust(wspace=0.2, hspace=0.6)



# workclass

sns.boxplot("workclass", "age", data=df, hue="income", ax=ax[0,0])

ax[0,0].tick_params(axis='x', labelrotation=90)

ax[0,0].legend(loc='upper right', facecolor='white')

ax[0,0].set_xlabel('')

ax[0,0].set_title("work class")



# education

sns.boxplot("education", "age", data=df, hue="income", ax=ax[0,1])

ax[0,1].tick_params(axis='x', labelrotation=90)

ax[0,1].legend(loc='upper right', facecolor='white')

ax[0,1].set_xlabel('')

ax[0,1].set_title("education")



# marital-status

sns.boxplot("marital_status", "age", data=df, hue="income", ax=ax[1,0])

ax[1,0].tick_params(axis='x', labelrotation=90)

ax[1,0].legend(loc='upper right', facecolor='white')

ax[1,0].set_xlabel('')

ax[1,0].set_title("marital-status")



# occupation

sns.boxplot("occupation", "age", data=df, hue="income", ax=ax[1,1])

ax[1,1].tick_params(axis='x', labelrotation=90)

ax[1,1].legend(loc='upper right', facecolor='white')

ax[1,1].set_xlabel('')

ax[1,1].set_title("occupation")



# relationship

sns.boxplot("relationship", "age", data=df, hue="income", ax=ax[2,0])

ax[2,0].tick_params(axis='x', labelrotation=90)

ax[2,0].legend(loc='upper right', facecolor='white')

ax[2,0].set_xlabel('')

ax[2,0].set_title("relationship")



# race

sns.boxplot("race", "age", data=df, hue="income", ax=ax[2,1])

ax[2,1].tick_params(axis='x', labelrotation=90)

ax[2,1].legend(loc='upper right', facecolor='white')

ax[2,1].set_xlabel('')

ax[2,1].set_title("race")



# sex

sns.boxplot("sex", "age", data=df, hue="income", ax=ax[3,0])

ax[3,0].tick_params(axis='x', labelrotation=90)

ax[3,0].legend(loc='upper right', facecolor='white')

ax[3,0].set_xlabel('')

ax[3,0].set_title("occupation")



# native-country

sns.boxplot("workclass", "age", data=df, hue="income", ax=ax[3,1])

ax[3,1].tick_params(axis='x', labelrotation=90)

ax[3,1].legend(loc='upper right', facecolor='white')

ax[3,1].set_xlabel('')

ax[3,1].set_title("native-country")
# histgram

fig, ax = plt.subplots(4,2, figsize=(25,30), facecolor="lavender")

plt.subplots_adjust(wspace=0.2, hspace=0.6)



# workclass

sns.boxplot("workclass", "capital_gain", data=df, hue="income", ax=ax[0,0])

ax[0,0].tick_params(axis='x', labelrotation=90)

ax[0,0].legend(loc='upper right', facecolor='white')

ax[0,0].set_xlabel('')

ax[0,0].set_yscale("log")

ax[0,0].set_title("work class")



# education

sns.boxplot("education", "capital_gain", data=df, hue="income", ax=ax[0,1])

ax[0,1].tick_params(axis='x', labelrotation=90)

ax[0,1].legend(loc='upper right', facecolor='white')

ax[0,1].set_xlabel('')

ax[0,1].set_yscale("log")

ax[0,1].set_title("education")



# marital-status

sns.boxplot("marital_status", "capital_gain", data=df, hue="income", ax=ax[1,0])

ax[1,0].tick_params(axis='x', labelrotation=90)

ax[1,0].legend(loc='upper right', facecolor='white')

ax[1,0].set_xlabel('')

ax[1,0].set_yscale("log")

ax[1,0].set_title("marital-status")



# occupation

sns.boxplot("occupation", "capital_gain", data=df, hue="income", ax=ax[1,1])

ax[1,1].tick_params(axis='x', labelrotation=90)

ax[1,1].legend(loc='upper right', facecolor='white')

ax[1,1].set_xlabel('')

ax[1,1].set_yscale("log")

ax[1,1].set_title("occupation")



# relationship

sns.boxplot("relationship", "capital_gain", data=df, hue="income", ax=ax[2,0])

ax[2,0].tick_params(axis='x', labelrotation=90)

ax[2,0].legend(loc='upper right', facecolor='white')

ax[2,0].set_xlabel('')

ax[2,0].set_yscale("log")

ax[2,0].set_title("relationship")



# race

sns.boxplot("race", "capital_gain", data=df, hue="income", ax=ax[2,1])

ax[2,1].tick_params(axis='x', labelrotation=90)

ax[2,1].legend(loc='upper right', facecolor='white')

ax[2,1].set_xlabel('')

ax[2,1].set_yscale("log")

ax[2,1].set_title("race")



# sex

sns.boxplot("sex", "capital_gain", data=df, hue="income", ax=ax[3,0])

ax[3,0].tick_params(axis='x', labelrotation=90)

ax[3,0].legend(loc='upper right', facecolor='white')

ax[3,0].set_xlabel('')

ax[3,0].set_yscale("log")

ax[3,0].set_title("occupation")



# native-country

sns.boxplot("workclass", "capital_gain", data=df, hue="income", ax=ax[3,1])

ax[3,1].tick_params(axis='x', labelrotation=90)

ax[3,1].legend(loc='upper right', facecolor='white')

ax[3,1].set_xlabel('')

ax[3,1].set_yscale("log")

ax[3,1].set_title("native-country")
# map dictionary

map_workclass = {" Private":0, " Self-emp-not-inc":1, " Local-gov":2, " ?":3, " State-gov":4, " Self-emp-inc":5, " Federal-gov":6, " Without-pay":7, " Never-worked":8}



map_education = {" HS-grad":0, " Some-college":1, " Bachelors":2, " Masters":3, " Assoc-voc":4, " 11th":5, " Assoc-acdm":6, " 10th":7,

                " 7th-8th":8, " Prof-school":9, " 9th":10, " 12th":10, " Doctorate":11, " 5th-6th":12, " 1st-4th":13, " Preschool":14}



map_marital_status = {" Married-civ-spouse":0, " Never-married":1, " Divorced":2, " Separated":3, " Widowed":4, " Married-spouse-absent":5, " Married-AF-spouse":6}



map_occupation = {" Prof-specialty":0, " Craft-repair":1, " Exec-managerial":2, " Adm-clerical":3, " Sales":4, " Other-service":5, " Machine-op-inspct":6, " ?":7,

                " Transport-moving":8, " Handlers-cleaners":9, " Farming-fishing":10, " Tech-support":10, " Protective-serv":11, " Priv-house-serv":12, " Armed-Forces":13}



map_relationship = {" Husband":0, " Not-in-family":1, " Own-child":2, " Unmarried":3, " Wife":4, " Other-relative":5}



map_race = {" White":0, " Black":1, " Asian-Pac-Islander":2, " Amer-Indian-Eskimo":3, " Other":4}



map_sex = {" Male":0, " Female":1}



# native-country are many categories, so i group the lower coutory and it is used creation of function and apply function.

def native_country_flg(x):

    if x["native_country"] == " United-States":

        res = 0

    elif x["native_country"] == " Mexico":

        res = 1

    elif x["native_country"] == " ?":

        res = 2

    elif x["native_country"] == " Philippines":

        res = 3

    else :

        res = 4

    return res



# mapping

df_class = df.copy()



df_class["workclass"] = df_class["workclass"].map(map_workclass)

df_class["education"] = df_class["education"].map(map_education)

df_class["marital_status"] = df_class["marital_status"].map(map_marital_status)

df_class["occupation"] = df_class["occupation"].map(map_occupation)

df_class["relationship"] = df_class["relationship"].map(map_relationship)

df_class["race"] = df_class["race"].map(map_race)

df_class["sex"] = df_class["sex"].map(map_sex)

df_class["native_country"] = df_class.apply(native_country_flg, axis=1)
# target variables, income

map_income = {" <=50K":0, " >50K":1}



df_class["income"] = df_class["income"].map(map_income)
df_class.head()
# Create Target variable and Explanatry variables

X = df_class.drop("income", axis=1)

y = df_class["income"]
# Data split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
# Difine class

forest = RandomForestClassifier(n_estimators=10, random_state=1)



# Gridsearch

param_range = [5,10,15,20]

leaf = [40,50, 60]

criterion = ["entropy", "gini", "error"]

param_grid = [{"n_estimators":param_range, "max_depth":param_range, "criterion":criterion, "max_leaf_nodes":leaf}]



gs = GridSearchCV(estimator=forest, param_grid=param_grid, scoring="accuracy", cv=10, n_jobs=1)



# Fitting

gs = gs.fit(X_train, y_train)



print(gs.best_score_)

print(gs.best_params_)
# Fitting for forest instance

forest = RandomForestClassifier(n_estimators=15, random_state=1,

                               criterion='gini', max_depth=15,

                               max_leaf_nodes=50)

forest.fit(X_train, y_train)



# Importance 

importance = forest.feature_importances_



# index

indices = np.argsort(importance)[::-1]



for f in range(X_train.shape[1]):

    print("%2d) %-*s %f" %(f+1, 30, X.columns[indices[f]], importance[indices[f]]))
# Visualization

plt.figure(figsize=(10,6), facecolor="lavender")



plt.title('Feature Importances')

plt.bar(range(X_train.shape[1]), importance[indices], color='blue', align='center')

plt.xticks(range(X_train.shape[1]), X.columns[indices], rotation=90)

plt.xlim([-1, X_train.shape[1]])

plt.tight_layout()

plt.show()
df_class["capital_value"] = df_class["capital_gain"] - df_class["capital_loss"]



df_class_4 = df_class[["marital_status", "capital_gain", "education_num", "relationship"]]

df_class_9 = df_class[["marital_status", "capital_value", "education_num", "relationship", "age", "occupation", "hours_per_week", "education"]]
# Data split

X_train_4, X_test_4, y_train_4, y_test_4 = train_test_split(df_class_4, y, test_size=0.3, random_state=1)



# Difine class

forest_4 = RandomForestClassifier(n_estimators=10, random_state=1)



# Gridsearch

param_range = [5,10,15,20]

leaf = [40,50,60]

criterion = ["entropy", "gini", "error"]

param_grid = [{"n_estimators":param_range, "max_depth":param_range, "criterion":criterion, "max_leaf_nodes":leaf}]



gs_4 = GridSearchCV(estimator=forest_4, param_grid=param_grid, scoring="accuracy", cv=10, n_jobs=1)



# Fitting

gs_4 = gs_4.fit(X_train_4, y_train_4)



print(gs_4.best_score_)

print(gs_4.best_params_)
# Data split

X_train_9, X_test_9, y_train_9, y_test_9 = train_test_split(df_class_9, y, test_size=0.3, random_state=1)



# Difine class

forest_9 = RandomForestClassifier(n_estimators=10, random_state=1)



# Gridsearch

param_range = [15,20, 25]

leaf = [60,70,80]

criterion = ["entropy", "gini", "error"]

param_grid = [{"n_estimators":param_range, "max_depth":param_range, "criterion":criterion, "max_leaf_nodes":leaf}]



gs_9 = GridSearchCV(estimator=forest_9, param_grid=param_grid, scoring="accuracy", cv=10, n_jobs=1)



# Fitting

gs_9 = gs_9.fit(X_train_9, y_train_9)



print(gs_9.best_score_)

print(gs_9.best_params_)
# original parameters

clf = gs.best_estimator_

print("-"*50)

print("Original parameters")



# Prediction

y_pred = clf.predict(X_test)

y_pred_train = clf.predict(X_train)



# Print prediction

print("confusion_matrix = \n", confusion_matrix(y_true=y_test, y_pred=y_pred))



print("*accuracy_train = %.3f" % accuracy_score(y_true=y_train, y_pred=y_pred_train))

print("accuracy = %.3f" % accuracy_score(y_true=y_test, y_pred=y_pred))



print("*precision_train = %.3f" % precision_score(y_true=y_train, y_pred=y_pred_train))

print("precision = %.3f" % precision_score(y_true=y_test, y_pred=y_pred))



print("*recall_train = %.3f" % recall_score(y_true=y_train, y_pred=y_pred_train))

print("recall = %.3f" % recall_score(y_true=y_test, y_pred=y_pred))



print("*f1_score_train = %.3f" % f1_score(y_true=y_train, y_pred=y_pred_train))

print("f1_score = %.3f" % f1_score(y_true=y_test, y_pred=y_pred))





# 1st parameters group (4 variables)

clf_4 = gs_4.best_estimator_

print("-"*50)

print("Original parameters")



# Prediction

y_pred = clf_4.predict(X_test_4)

y_pred_train = clf_4.predict(X_train_4)



# Print prediction

print("confusion_matrix = \n", confusion_matrix(y_true=y_test_4, y_pred=y_pred))



print("*accuracy_train = %.3f" % accuracy_score(y_true=y_train_4, y_pred=y_pred_train))

print("accuracy = %.3f" % accuracy_score(y_true=y_test_4, y_pred=y_pred))



print("*precision_train = %.3f" % precision_score(y_true=y_train_4, y_pred=y_pred_train))

print("precision = %.3f" % precision_score(y_true=y_test_4, y_pred=y_pred))



print("*recall_train = %.3f" % recall_score(y_true=y_train_4, y_pred=y_pred_train))

print("recall = %.3f" % recall_score(y_true=y_test_4, y_pred=y_pred))



print("*f1_score_train = %.3f" % f1_score(y_true=y_train_4, y_pred=y_pred_train))

print("f1_score = %.3f" % f1_score(y_true=y_test_4, y_pred=y_pred))





# 2nd parameters group (9 variables)

clf_9 = gs_9.best_estimator_

print("-"*50)

print("Original parameters")



# Prediction

y_pred = clf_9.predict(X_test_9)

y_pred_train = clf_9.predict(X_train_9)



# Print prediction

print("confusion_matrix = \n", confusion_matrix(y_true=y_test_9, y_pred=y_pred))



print("*accuracy_train = %.3f" % accuracy_score(y_true=y_train_9, y_pred=y_pred_train))

print("accuracy = %.3f" % accuracy_score(y_true=y_test_9, y_pred=y_pred))



print("*precision_train = %.3f" % precision_score(y_true=y_train_9, y_pred=y_pred_train))

print("precision = %.3f" % precision_score(y_true=y_test_9, y_pred=y_pred))



print("*recall_train = %.3f" % recall_score(y_true=y_train_9, y_pred=y_pred_train))

print("recall = %.3f" % recall_score(y_true=y_test_9, y_pred=y_pred))



print("*f1_score_train = %.3f" % f1_score(y_true=y_train_9, y_pred=y_pred_train))

print("f1_score = %.3f" % f1_score(y_true=y_test_9, y_pred=y_pred))
# age_cate

age_qua25=df_class["age"].quantile(0.25)

age_qua50=df_class["age"].quantile(0.5)

age_qua75=df_class["age"].quantile(0.75)

def age_cate(x):

    if x["age"] < age_qua25:

        res = 0

    elif x["age"] < age_qua50 and x["age"] >= age_qua25:

        res = 1

    elif x["age"] < age_qua75 and x["age"] >= age_qua50:

        res = 2

    else:

        res = 3

    return res



df_class["age"] = df_class.apply(age_cate, axis=1)



# fnlwgt_cate

fnl_qua25=df_class["fnlwgt"].quantile(0.25)

fnl_qua50=df_class["fnlwgt"].quantile(0.5)

fnl_qua75=df_class["fnlwgt"].quantile(0.75)

def fnlwgt_cate(x):

    if x["fnlwgt"] < fnl_qua25:

        res = 0

    elif x["fnlwgt"] < fnl_qua50 and x["fnlwgt"] >= fnl_qua25:

        res = 1

    elif x["fnlwgt"] < fnl_qua75 and x["fnlwgt"] >= fnl_qua50:

        res = 2

    else:

        res = 3

    return res  



df_class["fnlwgt"] = df_class.apply(fnlwgt_cate, axis=1)



# capital-value_cate

cav_qua25=df_class["capital_value"].quantile(0.25)

cav_qua50=df_class["capital_value"].quantile(0.5)

cav_qua75=df_class["capital_value"].quantile(0.75)

def capital_value_cate(x):

    if x["capital_value"] < cav_qua25:

        res = 0

    elif x["capital_value"] < cav_qua50 and x["capital_value"] >= cav_qua25:

        res = 1

    elif x["capital_value"] < cav_qua75 and x["capital_value"] >= cav_qua50:

        res = 2

    else:

        res = 3

    return res 



df_class["capital_value"] = df_class.apply(capital_value_cate, axis=1)



# education-num_cate

edu_qua25=df_class["education_num"].quantile(0.25)

edu_qua50=df_class["education_num"].quantile(0.5)

edu_qua75=df_class["education_num"].quantile(0.75)

def education_num_cate(x):

    if x["education_num"] < edu_qua25:

        res = 0

    elif x["education_num"] < edu_qua50 and x["education_num"] >= edu_qua25:

        res = 1

    elif x["education_num"] < edu_qua75 and x["education_num"] >= edu_qua50:

        res = 2

    else:

        res = 3

    return res



df_class["education_num"] = df_class.apply(education_num_cate, axis=1)



# hours-per-week

hou_qua25=df_class["hours_per_week"].quantile(0.25)

hou_qua50=df_class["hours_per_week"].quantile(0.5)

hou_qua75=df_class["hours_per_week"].quantile(0.75)

def hours_per_week_cate(x):

    if x["hours_per_week"] < hou_qua25:

        res = 0

    elif x["hours_per_week"] < hou_qua50 and x["hours_per_week"] >= hou_qua25:

        res = 1

    elif x["hours_per_week"] < hou_qua75 and x["hours_per_week"] >= hou_qua50:

        res = 2

    else:

        res = 3

    return res



df_class["hours_per_week"] = df_class.apply(hours_per_week_cate, axis=1)
df_9_cate = df_class[["marital_status", "capital_value", "education_num", "relationship", "age", "occupation", "hours_per_week", "education", "income"]]
# Create Target variable and Explanatry variables

X = df_9_cate.drop("income", axis=1)

y = df_9_cate["income"]



# Data split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
# Difine class

forest = RandomForestClassifier(n_estimators=10, random_state=1)



# Gridsearch

param_range = [5,10,15,20]

leaf = [60,70, 80]

criterion = ["entropy", "gini", "error"]

param_grid = [{"n_estimators":param_range, "max_depth":param_range, "criterion":criterion, "max_leaf_nodes":leaf}]



gs = GridSearchCV(estimator=forest, param_grid=param_grid, scoring="accuracy", cv=10, n_jobs=1)



# Fitting

gs = gs.fit(X_train, y_train)



print(gs.best_score_)

print(gs.best_params_)
# original parameters

clf = gs.best_estimator_

print("-"*50)

print("Original parameters")



# Prediction

y_pred = clf.predict(X_test)

y_pred_train = clf.predict(X_train)



# Print prediction

print("confusion_matrix = \n", confusion_matrix(y_true=y_test, y_pred=y_pred))



print("*accuracy_train = %.3f" % accuracy_score(y_true=y_train, y_pred=y_pred_train))

print("accuracy = %.3f" % accuracy_score(y_true=y_test, y_pred=y_pred))



print("*precision_train = %.3f" % precision_score(y_true=y_train, y_pred=y_pred_train))

print("precision = %.3f" % precision_score(y_true=y_test, y_pred=y_pred))



print("*recall_train = %.3f" % recall_score(y_true=y_train, y_pred=y_pred_train))

print("recall = %.3f" % recall_score(y_true=y_test, y_pred=y_pred))



print("*f1_score_train = %.3f" % f1_score(y_true=y_train, y_pred=y_pred_train))

print("f1_score = %.3f" % f1_score(y_true=y_test, y_pred=y_pred))