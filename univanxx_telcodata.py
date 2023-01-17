import numpy as np
import pandas as pd
import os
Data = pd.read_csv('/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv',usecols = lambda column : column not in 
["customerID"])
Data.info()
Data.head()
def display_missing(df):    
    for col in df.columns.tolist():          
        print('{} column missing values: {}'.format(col, df[col].isnull().sum()))
    print('\n')
    
display_missing(Data)
columns_to_choose = [cname for cname in Data.columns if cname != 'Churn']
  
indexNames = Data[Data['TotalCharges'] == " "].index
print(indexNames)
Data.drop(index = indexNames,inplace = True)

Data['TotalCharges'] = pd.to_numeric(Data['TotalCharges'])

X = pd.DataFrame(Data, columns=columns_to_choose)
y = Data.Churn     
y.head()
v_0 = 0
v_1 = 0
for i in range(len(y)):
    if y.iloc[i] == 'Yes':
        v_1 += 1
    else:
        v_0 += 1
        
print(v_0)
print(v_1)
from sklearn.model_selection import train_test_split

X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                                random_state=0)
# "Cardinality" means the number of unique values in a column
low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and 
                        X_train_full[cname].dtype == "object"]
# Select numeric columns
numeric_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]
# Keep selected columns only
my_cols = low_cardinality_cols + numeric_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_train = pd.get_dummies(X_train)
X_valid = pd.get_dummies(X_valid)
X_train, X_valid = X_train.align(X_valid, join='left', axis=1)  # левый join по столбцам
X_train.head()
'''
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier


# Define the model
my_model = XGBClassifier()

xgb_hyperparams = {'n_estimators': [1250,1500,1750,2000],
                   'learning_rate': [i/100 for i in range(15,66,5)],
                  'max_depth' : [4,5,6]}


grid_search = GridSearchCV(my_model,xgb_hyperparams,cv=5)

fit_params = {'early_stopping_rounds' : 5, 
              'eval_set' : [(X_valid, y_valid)], 
              'verbose' : False}


# Fit the model
grid_search.fit(X_train, y_train,**fit_params)

print(grid_search.best_params_)

print(grid_search.best_estimator_)
'''
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

from xgboost import XGBClassifier

my_model = XGBClassifier(learning_rate= 0.4, n_estimators = 1250,max_depth = 4)
my_model.fit(X_train, y_train)

predictions = my_model.predict(X_valid)

y_valid = pd.get_dummies(y_valid)
predictions = pd.get_dummies(predictions)

# Calculate accuracy
ACC = accuracy_score(y_valid, predictions)
F1 = f1_score(y_valid, predictions, average='weighted')

print("Accuracy:" , ACC)
print("F1:" , F1)
from sklearn.metrics import classification_report

report = classification_report(y_valid, predictions, target_names=['Non-churned', 'Churned'])
print(report)
from sklearn.metrics import roc_auc_score

predictions = pd.get_dummies(predictions)
y_valid = pd.get_dummies(y_valid)

roc_auc_score(predictions, y_valid)
Data = pd.read_csv('/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv',usecols = lambda column : column not in 
["customerID"])

columns_to_choose = [cname for cname in Data.columns if cname != 'Churn']  
indexNames = Data[Data['TotalCharges'] == " "].index
print(indexNames)
Data.drop(index = indexNames,inplace = True)
Data['TotalCharges'] = pd.to_numeric(Data['TotalCharges'])
X = pd.DataFrame(Data, columns=columns_to_choose)
y = Data.Churn
numeric_cols = ['tenure','MonthlyCharges','TotalCharges']
categorial_cols = [cname for cname in X.columns if cname not in numeric_cols]
for cname in categorial_cols:
        print('Number of {} column unique values: {}'.format(cname, len(pd.unique(X[cname]))))
from sklearn.model_selection import train_test_split

X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                                random_state=0)
# Creating the count encoder
import category_encoders as ce

count_enc = ce.CountEncoder(cols=categorial_cols,normalize=True)
count_enc.fit(X_train_full[categorial_cols])
X_train_numeric = X_train_full[numeric_cols]
X_valid_numeric = X_valid_full[numeric_cols]
train_encoded = X_train_numeric.join(count_enc.transform(X_train_full[categorial_cols]).add_suffix('_count'))
valid_encoded = X_valid_numeric.join(count_enc.transform(X_valid_full[categorial_cols]).add_suffix('_count'))
#train_encoded['TotalCharges'] = pd.to_numeric(train_encoded['TotalCharges'])
#valid_encoded['TotalCharges'] = pd.to_numeric(valid_encoded['TotalCharges'])
from sklearn.metrics import accuracy_score

my_model = XGBClassifier(learning_rate= 0.4, n_estimators = 1250,max_depth = 4)
my_model.fit(train_encoded, y_train)

predictions = my_model.predict(valid_encoded)

y_valid = pd.get_dummies(y_valid)
predictions = pd.get_dummies(predictions)

# Calculate accuracy
ACC = accuracy_score(predictions, y_valid)
F1 = f1_score(y_valid, predictions, average='weighted')

print("Accuracy:" , ACC)
print("F1:" , F1)
from sklearn.metrics import roc_auc_score

predictions = pd.get_dummies(predictions)
y_valid = pd.get_dummies(y_valid)

roc_auc_score(predictions, y_valid)
X = pd.DataFrame(Data, columns=columns_to_choose)
y = Data.Churn
numeric_cols = ['tenure','MonthlyCharges','TotalCharges']
categorial_cols = [cname for cname in X.columns if cname not in numeric_cols and len(pd.unique(X[cname])) > 2]
cols_2 = [cname for cname in X.columns if cname not in numeric_cols and cname not in categorial_cols]
from sklearn.model_selection import train_test_split

X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                                random_state=0)
# Creating the count encoder
import category_encoders as ce

count_enc = ce.CountEncoder(cols=categorial_cols)
count_enc.fit(X_train_full[categorial_cols])

X_train_cols_2 = X_train_full[cols_2].copy()
X_train_cols_2 = pd.get_dummies(X_train_cols_2)

X_valid_cols_2 = X_valid_full[cols_2].copy()
X_valid_cols_2 = pd.get_dummies(X_valid_cols_2)
X_train_numeric = X_train_full[numeric_cols].copy()
X_valid_numeric = X_valid_full[numeric_cols].copy()
train_encoded = X_train_numeric.join(count_enc.transform(X_train_full[categorial_cols]).add_suffix('_count'))
valid_encoded = X_valid_numeric.join(count_enc.transform(X_valid_full[categorial_cols]).add_suffix('_count'))
train_encoded = train_encoded.join(X_train_cols_2)
valid_encoded = valid_encoded.join(X_valid_cols_2)
#train_encoded['TotalCharges'] = pd.to_numeric(train_encoded['TotalCharges'])
#valid_encoded['TotalCharges'] = pd.to_numeric(valid_encoded['TotalCharges'])
train_encoded.head()
from sklearn.metrics import accuracy_score

my_model = XGBClassifier(learning_rate= 0.4, n_estimators = 1250,max_depth = 4)
my_model.fit(train_encoded, y_train)

predictions = my_model.predict(valid_encoded)

y_valid = pd.get_dummies(y_valid)
predictions = pd.get_dummies(predictions)

# Calculate accuracy
ACC = accuracy_score(predictions, y_valid)
F1 = f1_score(y_valid, predictions, average='weighted')

print("Accuracy:" , ACC)
print("F1:" , F1)
X = pd.DataFrame(Data, columns=columns_to_choose)
y = Data.Churn
from sklearn.model_selection import train_test_split

X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                                random_state=0)
numeric_cols = ['tenure','MonthlyCharges','TotalCharges']
categorial_cols = [cname for cname in X.columns if cname not in numeric_cols]
X_train_cat = X_train_full[categorial_cols].copy()
X_train_cat = pd.get_dummies(X_train_cat)

X_valid_cat = X_valid_full[categorial_cols].copy()
X_valid_cat = pd.get_dummies(X_valid_cat)

X_train_numeric = X_train_full[numeric_cols].copy()
X_valid_numeric = X_valid_full[numeric_cols].copy()

train_encoded = X_train_numeric.join(X_train_cat)
valid_encoded = X_valid_numeric.join(X_valid_cat)
train_encoded.head()
import itertools
from sklearn import preprocessing

interactions_train = pd.DataFrame(index=X_train_full.index)
interactions_valid = pd.DataFrame(index=X_valid_full.index)

# Iterate through each pair of features, combine them into interaction features
for i in itertools.combinations(categorial_cols,2):
    
    named = (i[0] + "_" + i[1])
    
    mixed_cat = X_train_full[i[0]].map(str) + "_" + X_train_full[i[1]].map(str)
    label_enc = preprocessing.LabelEncoder()
    interactions_train[named] = label_enc.fit_transform(mixed_cat)
    
    mixed_cat = X_valid_full[i[0]].map(str) + "_" + X_valid_full[i[1]].map(str)
    label_enc = preprocessing.LabelEncoder()
    interactions_valid[named] = label_enc.fit_transform(mixed_cat)  
interactions_train.head()
train_encoded = train_encoded.join(interactions_train)
valid_encoded = valid_encoded.join(interactions_valid)
train_encoded.head()
valid_encoded.head()
from sklearn.metrics import accuracy_score

my_model = XGBClassifier(learning_rate= 0.4, n_estimators = 1250,max_depth = 4)
my_model.fit(train_encoded, y_train)

predictions = my_model.predict(valid_encoded)

y_valid = pd.get_dummies(y_valid)
predictions = pd.get_dummies(predictions)

# Calculate accuracy
ACC = accuracy_score(predictions, y_valid)
F1 = f1_score(y_valid, predictions, average='weighted')

print("Accuracy:" , ACC)
print("F1:" , F1)
from sklearn.metrics import roc_auc_score

predictions = pd.get_dummies(predictions)
y_valid = pd.get_dummies(y_valid)

roc_auc_score(predictions, y_valid)
X = pd.DataFrame(Data, columns=columns_to_choose)
y = Data.Churn
from sklearn.model_selection import train_test_split

X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                                random_state=0)
numeric_cols = ['tenure','MonthlyCharges','TotalCharges']
categorial_cols = [cname for cname in X.columns if cname not in numeric_cols]
# Keep selected columns only
X_train = X_train_full[categorial_cols].copy()
X_valid = X_valid_full[categorial_cols].copy()

X_train = pd.get_dummies(X_train)
X_valid = pd.get_dummies(X_valid)

train_encoded = X_train.join(X_train_full[numeric_cols])
valid_encoded = X_valid.join(X_valid_full[numeric_cols])

# X_train, X_valid = X_train.align(X_valid, join='left', axis=1)  # левый join по столбцам
interactions_train = pd.DataFrame(index=X_train_full.index)
interactions_valid = pd.DataFrame(index=X_valid_full.index)

# Iterate through each pair of features, combine them into interaction features
for i in itertools.combinations(categorial_cols,2):
    
    named = (i[0] + "_" + i[1])
    
    mixed_cat = X_train_full[i[0]].map(str) + "_" + X_train_full[i[1]].map(str)
    label_enc = preprocessing.LabelEncoder()
    interactions_train[named] = label_enc.fit_transform(mixed_cat)
    
    mixed_cat = X_valid_full[i[0]].map(str) + "_" + X_valid_full[i[1]].map(str)
    label_enc = preprocessing.LabelEncoder()
    interactions_valid[named] = label_enc.fit_transform(mixed_cat)
train_encoded = train_encoded.join(interactions_train)
valid_encoded = valid_encoded.join(interactions_valid)
from sklearn.metrics import accuracy_score

my_model = XGBClassifier(learning_rate= 0.4, n_estimators = 1250,max_depth = 4)
my_model.fit(train_encoded, y_train)

predictions = my_model.predict(valid_encoded)

y_valid = pd.get_dummies(y_valid)
predictions = pd.get_dummies(predictions)

# Calculate accuracy
ACC = accuracy_score(predictions, y_valid)
F1 = f1_score(y_valid, predictions, average='weighted')

print("Accuracy:" , ACC)
print("F1:" , F1)
from sklearn.metrics import classification_report

report = classification_report(y_valid, predictions, target_names=['Non-churned', 'Churned'])
print(report)
valid_encoded
train_encoded
