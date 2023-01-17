import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
train_data_identity = pd.read_csv('../input/ieee-fraud-detection/train_identity.csv')
train_data_transaction = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv')
test_data_identity = pd.read_csv('../input/ieee-fraud-detection/test_identity.csv')
test_data_transaction = pd.read_csv('../input/ieee-fraud-detection/test_transaction.csv')
del train_data_identity
del train_data_transaction
del test_data_identity
del test_data_transaction
import gc
gc.collect()
train_data_identity.head()
pd.set_option('display.max_columns', None)
train_data_transaction.head()
# combining the datasets
combined_train_dataset = train_data_transaction.merge(train_data_identity, how='inner',on='TransactionID') # I know this drops like 400000 rows...
len(combined_train_dataset['TransactionID'])
combined_train_dataset.head()
columnsToDelete = []
for col in combined_train_dataset.columns:
    if(combined_train_dataset[col].isnull().sum()/len(combined_train_dataset[col]) >= 0.8):
        print(col, "% NaN:", combined_train_dataset[col].isnull().sum()/len(combined_train_dataset[col]))
        columnsToDelete.append(col)
combinedToDelete = list(set(columnsToDelete + cols_not_in_test))
combined_train_dataset = combined_train_dataset.drop(columns=combinedToDelete)
len(combined_train_dataset.columns)
import missingno as msno
msno.matrix(combined_train_dataset.iloc[:,:20],labels=True,fontsize=10)
combined_train_dataset.iloc[:,:2]
# Already removed all columns with 80% or more NaN values, now I guess I'll just chuck vals into an imputer and see where it goes...
numericalCols = []
categoricalCols = []

for col in combined_train_dataset.columns:
    if(combined_train_dataset[col].dtype == 'object'):
        categoricalCols.append(col)
    else:
        numericalCols.append(col)
numericalCols.remove('isFraud')
X = combined_train_dataset.drop(columns=['isFraud'])
y = combined_train_dataset['isFraud']
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numericalCols),
        ('cat', categorical_transformer, categoricalCols)
    ])
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25)
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
'''
KNeighborsClassifier(3),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB()'''
classifiers = [
    GradientBoostingClassifier(random_state=0)
    ]
for cls in classifiers:
    # Bundle preprocessing and modeling code in a pipeline
    my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('model', cls)
                                 ])

    # Preprocessing of training data, fit model 
    print("Training" , cls)
    my_pipeline.fit(X_train, y_train)

    # Preprocessing of validation data, get predictions
    preds = my_pipeline.predict(X_test)

    # Evaluate the model
    score = accuracy_score(y_test, preds)
    print('Accuracy of',cls,":", score)
# KNeighborsClassifier(3) : 0.9064865
# DecisionTreeClassifier(max_depth=5) : 0.953
# RandomForestClassifier(max_depth=5, max_features=1, n_estimators=10) : 0.9202695582240218
# MLPClassifier(alpha=1, max_iter=1000) : 0.7159655009844976
# AdaBoostClassifier() : 0.9548240383815414
# GaussianNB() : 0.1614298788097285
# GradientBoostingClassifier(random_state=0) : 0.9610915444133226
len(test_data_identity['TransactionID'])
len(test_data_transaction['TransactionID'])
test_data_identity_cpy = test_data_identity
test_data_transaction_cpy = test_data_transaction
test_data_transaction_cpy.merge(test_data_identity_cpy, how='outer',on='TransactionID')
missingCols = ['id_07','id_08', 'id_21' ,'id_22', 'id_23' ,'id_24' ,'id_25' ,'id_26', 'id_27']
columnsToDelete_cpy = columnsToDelete
for i in missingCols:
    try:
        columnsToDelete_cpy.remove(i)
    except:
        print(i, "not in list")
test_data_transaction_cpy = test_data_transaction_cpy.drop(columns=columnsToDelete_cpy)
len(test_data_transaction_cpy.columns)
len(X_test.columns)
cols_not_in_test = []
for i in train_data_identity.columns:
    if i not in test_data_transaction_cpy.columns:
        cols_not_in_test.append(i)
cols_not_in_test
len(combined_train_dataset.columns)
len(test_data_transaction_cpy.columns)
# What cols are in test that are not in train and vice versa?
columnsToDelete = []
for col in combined_train_dataset:
    if col not in test_data_transaction_cpy.columns:
        columnsToDelete.append(col)
for col in test_data_transaction_cpy:
    if col not in combined_train_dataset.columns:
        columnsToDelete.append(col)
columnsToDelete.remove('isFraud')
combined_train_dataset = combined_train_dataset.drop(columns=columnsToDelete,errors='ignore')
test_data_transaction_cpy = test_data_transaction_cpy.drop(columns=columnsToDelete,errors='ignore')
len(combined_train_dataset.columns)
len(test_data_transaction_cpy.columns)
predictions = my_pipeline.predict(test_data_transaction_cpy)
transactionIDs = test_data_transaction_cpy['TransactionID'].values
submit_df = pd.DataFrame(data=transactionIDs,columns=['TransactionID'])
submit_df['isFraud'] = predictions
submit_df = submit_df.set_index('TransactionID')
submit_df['isFraud'].value_counts()
submit_df.to_csv('submission_v1.csv')
