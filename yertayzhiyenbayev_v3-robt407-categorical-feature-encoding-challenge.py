import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt # for plots

import seaborn as sns



from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV



from sklearn.linear_model import LogisticRegression



from sklearn.preprocessing import OneHotEncoder

one_hot = OneHotEncoder()



from sklearn.preprocessing import LabelEncoder

label_enc = LabelEncoder()



from sklearn.feature_extraction import FeatureHasher

feature_hashing = FeatureHasher(input_type='string')



'''

 Use Logistic Regression and return the accuracy score

'''

def exec_logistic(X,y, Max_iter):

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0, test_size = 1/5)

    

    log_reg = LogisticRegression(max_iter = Max_iter)

    log_reg.fit(X_train, y_train)

    y_pred = log_reg.predict(X_test)

    

    print('Accuracy score: ', accuracy_score(y_test, y_pred))
d_train = pd.read_csv("/kaggle/input/cat-in-the-dat/train.csv")

d_test = pd.read_csv("/kaggle/input/cat-in-the-dat/test.csv")
d_train.info()
d_train.head()
print('Number of missing values in the train set:', d_train.isna().sum().sum())

print('Number of missing values in the test set:', d_test.isna().sum().sum())
sns.set()

sns.pairplot(d_train, height = 2.5)

plt.show()
X = d_train.drop(['target'],axis=1)



array_objects = [] # Store the names of the columns of type 'o'

array_integers = [] # Store the names of the columns of integer type 



for col in zip(X.columns):

  if(X[col[0]].dtype == 'object'):

    array_objects.append(col[0])

  else:

    array_integers.append(col[0])
X[array_integers].head()
X[array_objects].head()
# Label encoding of the columns of type 'object'

for col in zip(array_objects):

  X[col[0]] = label_enc.fit_transform(X[col[0]])



X
%%time



exec_logistic(X, d_train['target'], 100)
X = d_train.drop(['target'],axis = 1)

X.shape
X = one_hot.fit_transform(X)



X.shape
%%time



exec_logistic(X, d_train['target'], 1000)
X = d_train.drop(['target'],axis = 1)



# Label encoding of the columns of type 'object'

for col in zip(array_objects):

  X[col[0]] = label_enc.fit_transform(X[col[0]])



nom_2_one_hot = one_hot.fit_transform(X.nom_2.values.reshape(-1,1)).toarray()

nom_3_one_hot = one_hot.fit_transform(X.nom_3.values.reshape(-1,1)).toarray()

nom_4_one_hot = one_hot.fit_transform(X.nom_4.values.reshape(-1,1)).toarray()



day_one_hot = one_hot.fit_transform(X.day.values.reshape(-1,1)).toarray()

month_one_hot = one_hot.fit_transform(X.month.values.reshape(-1,1)).toarray()
df_nom_2 = pd.DataFrame(nom_2_one_hot, columns = ["nom2_"+str(int(i)) for i in range(nom_2_one_hot.shape[1])])

X = X.drop(['nom_2'],axis = 1)

X = pd.concat([X, df_nom_2], axis = 1)



df_nom_3 = pd.DataFrame(nom_3_one_hot, columns = ["nom3_"+str(int(i)) for i in range(nom_3_one_hot.shape[1])])

X = X.drop(['nom_3'],axis=1)

X = pd.concat([X, df_nom_3], axis=1)



df_nom_4 = pd.DataFrame(nom_4_one_hot, columns = ["nom4_"+str(int(i)) for i in range(nom_4_one_hot.shape[1])])

X = X.drop(['nom_4'],axis=1)

X = pd.concat([X, df_nom_4], axis=1)



df_day = pd.DataFrame(day_one_hot, columns = ["day_"+str(int(i)) for i in range(day_one_hot.shape[1])])

X = X.drop(['day'],axis=1)

X = pd.concat([X, df_day], axis=1)



df_month = pd.DataFrame(month_one_hot, columns = ["month_"+str(int(i)) for i in range(month_one_hot.shape[1])])

X = X.drop(['month'],axis=1)

X = pd.concat([X, df_month], axis=1)
X
%%time



exec_logistic(X, d_train['target'], 1000)
X = d_train.drop(['target'],axis = 1)
%%time



X_hash = X.copy()

for c in X.columns:

    X_hash[c] = X[c].astype('str')      



X_feature_hashed = feature_hashing.transform(X_hash.values)
%%time



exec_logistic(X_feature_hashed, d_train['target'], 2000)
X = d_train

X_fold = X.copy()



to_add = ['ord_0','day','month']



for add in zip(to_add):

  array_objects.append(add[0])



X_fold['bin_3'] = label_enc.fit_transform(X_fold['bin_3'])

X_fold['bin_4'] = label_enc.fit_transform(X_fold['bin_4'])

X_fold[array_objects] = X_fold[array_objects].astype('object')
%%time



kf = KFold(n_splits = 5, shuffle = False, random_state = 0)

for train_ind,val_ind in kf.split(X):

    for col in zip(array_objects):

        replaced=dict(X.iloc[train_ind][[col[0],'target']].groupby(col[0])['target'].mean())

        X_fold.loc[val_ind,col[0]]=X_fold.iloc[val_ind][col[0]].replace(replaced).values
X_fold.head()
%%time



X_hash = X.copy()

for c in X.columns:

    X_hash[c] = X[c].astype('str')      



X_feature_hashed = feature_hashing.transform(X_hash.values)
%%time



exec_logistic(X_feature_hashed, d_train['target'], 2000)
%%time



linear_param_grid = {'C': np.array([1e-2, 1e-1, 1, 4]),

                     'solver': ['newton-cg', 'lbfgs', 'saga'],

                     'class_weight': ['balanced', None]}



log_reg = LogisticRegression(max_iter = 2000)



clf = GridSearchCV(log_reg, linear_param_grid , cv = 3, n_jobs = -1, scoring = 'roc_auc', verbose = 1)



best_model = clf.fit(X_feature_hashed, d_train['target'])



print(best_model.best_estimator_)

print(best_model.best_score_)

print(best_model.best_params_)