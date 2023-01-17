import pandas as pd

import numpy as np

from pandas.api.types import is_string_dtype, is_numeric_dtype

from sklearn import preprocessing

from sklearn.model_selection import GridSearchCV

from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, matthews_corrcoef, f1_score

import matplotlib.pyplot as plt

import seaborn as sns

import re



%matplotlib inline
# Assign current path to c_path

c_path = '../input'
my_df = pd.read_csv(f'{c_path}/adult.csv')



print(my_df.shape)

my_df.head()
def mydf_splitter(my_df, num_rows):

    return my_df[:num_rows].copy(), my_df[num_rows:].copy()



mydf_train_valid, mydf_test = mydf_splitter(my_df, 30000)



print(mydf_train_valid.shape, mydf_test.shape)
# Get a general picture of mydf_train_valid

mydf_train_valid.info()
'''Convert data in columns from 'object' type to 'category' type'''



def str_to_cat(my_df):

    for p, q in my_df.items(): 

        if is_string_dtype(q): 

            my_df[p] = q.astype('category').cat.as_ordered()

    return my_df



mydf_train_valid = str_to_cat(mydf_train_valid)



mydf_train_valid.info()
# Check information of some 'category' type columns

print(mydf_train_valid["workclass"].cat.categories)

print(mydf_train_valid["education"].cat.categories)

mydf_train_valid.head(3)
'''Convert data in these 'category' type culumns to their corresponding numerical values'''



def cat_to_num(my_df):

    for p, q in my_df.items():

        if not is_numeric_dtype(q):

            my_df[p] = q.cat.codes

    return my_df

            

mydf_train_valid = cat_to_num(mydf_train_valid)



mydf_train_valid.head()
"""Seperate data into the X and Y varibles"""



Y_train_valid = mydf_train_valid["income"]

X_train_valid = mydf_train_valid.drop(["income"], axis=1)



print(X_train_valid.shape, Y_train_valid.shape)



Y_train_valid.unique()
"""X variable columns might be a continuous variable column or a categorical 

variable column. Seperate into continuous variables and categorical variables"""



X_train_valid_cat = X_train_valid[["workclass", "education", "marital.status", "occupation", 

                                   "relationship", "race", "sex", "native.country"]]

X_train_valid_con = X_train_valid.drop(X_train_valid_cat, axis=1)



print(X_train_valid_cat.shape, X_train_valid_con.shape)
'''Scale the continuous variables. To standardize (includes scaling), 

we subtract mean of that column from every value, then divide the results 

by the variable's standard deviation'''



scaler = preprocessing.StandardScaler().fit(X_train_valid_con)

X_train_valid_con_sc = pd.DataFrame(scaler.transform(X_train_valid_con))

X_train_valid_con_sc.columns = ["age","fnlwgt", "education.num", "capital.gain", "capital.loss", "hours.per.week"]



print(X_train_valid_con_sc.shape)

X_train_valid_con_sc.head()
df_list = [X_train_valid_cat, X_train_valid_con_sc]

X_full = pd.concat(df_list, axis = 1)



print(X_full.shape)

X_full.head()
X_train, X_valid = mydf_splitter(X_full, 27500)

Y_train, Y_valid = mydf_splitter(Y_train_valid, 27500)



print(X_train.shape, X_valid.shape, Y_train.shape, Y_valid.shape)
"""Use multiple for-loops to search for the best combination of parameters for kNN"""



params = {'n_neighbors': [k for k in range(1, 20, 2)],

        'weights': ['uniform', 'distance'],

        'metric': ['manhattan', 'euclidean']}



num_neighs = list()

val_weights = list()

val_metric = list()

accuracy_list = list()



for n in params["n_neighbors"]:

    for w in params["weights"]:

        for m in params["metric"]:

                my_knn_model = KNeighborsClassifier(n_neighbors=n, weights=w, metric=m)

                my_knn_model.fit(X_train, Y_train)

                Y_pred = my_knn_model.predict(X_valid)

                accuracy = accuracy_score(Y_valid, Y_pred)

                num_neighs.append(n)

                val_weights.append(w)

                val_metric.append(m)

                accuracy_list.append(accuracy)

            
eval_df =  pd.DataFrame({"n_neighbors": num_neighs, "weights": val_weights, 

                         "metric": val_metric, "accuracy score": accuracy_list})

eval_df.index = eval_df.index + 1

eval_df.index.name = "No."



eval_df
#Plot accuracy Vs No.

plt.figure(figsize=(8, 5), dpi=80)

plt.xticks(np.arange(1,  41,  1))

plt.scatter(eval_df.index, eval_df["accuracy score"], marker='+')
"""We then use the best combination of parameters- 

{n_neighbors=19, weights='uniform', metric='euclidean'} to train our kNN model""" 



my_knn_model_final = KNeighborsClassifier(n_neighbors=19, weights='uniform', metric='euclidean')

my_knn_model_final.fit(X_full, Y_train_valid)
'''Before we can apply our kNN model on the test set, we

need to preprocess the test set in exactly the same way we did the

train-valid set'''



print(mydf_test.shape)

mydf_test.head()
mydf_test = str_to_cat(mydf_test)



mydf_test.info()
mydf_test = cat_to_num(mydf_test)



mydf_test.head()
Y_test = mydf_test["income"]

X_test = mydf_test.drop(["income"], axis=1)



print(X_test.shape, Y_test.shape)
X_test_cat = X_test[["workclass", "education", "marital.status", "occupation", 

                                   "relationship", "race", "sex", "native.country"]]

X_test_con = X_test.drop(X_test_cat, axis=1)



print(X_test_cat.shape, X_test_con.shape)
scaler = preprocessing.StandardScaler().fit(X_test_con)

X_test_con_sc = pd.DataFrame(scaler.transform(X_test_con))

X_test_con_sc.columns = ["age","fnlwgt", "education.num", "capital.gain", "capital.loss", "hours.per.week"]



print(X_test_con_sc.shape)

X_test_con_sc.head()
X_test_cat.index = [i for i in range(len(X_test_cat))]

df_list = [X_test_cat, X_test_con_sc]

X_test = pd.concat(df_list, axis = 1)

X_test.index = range(30000, len(X_test)+30000) 



print(X_test.shape)

X_test.head()
# Testing...

Y_test_pred = my_knn_model_final.predict(X_test)



print(accuracy_score(Y_test, Y_test_pred),

      matthews_corrcoef(Y_test,Y_test_pred), f1_score(Y_test,Y_test_pred))
my_knn_cmatrix = confusion_matrix(Y_test, Y_test_pred)



my_knn_df = pd.DataFrame(my_knn_cmatrix)

plt.figure(figsize = (8, 8))

sns.heatmap(my_knn_df, xticklabels = ["<=50K",">50K"],

            yticklabels = ["<=50K",">50K"], annot = True)