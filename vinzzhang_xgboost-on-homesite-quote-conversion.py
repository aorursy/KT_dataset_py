import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
train = pd.read_csv('../input/train.csv')
train.head()
test = pd.read_csv('../input/test.csv')
test.head()
# Drop meanliness feature: 'QuoteNumber'
train = train.drop('QuoteNumber',axis=1)
test = test.drop('QuoteNumber', axis=1)
# Convert str_type 'Date' into date_type 
train['Date'] = pd.to_datetime(train['Original_Quote_Date'])
test['Date'] = pd.to_datetime(test['Original_Quote_Date'])

# Drop 'Original_Quote_date'
train = train.drop('Original_Quote_Date',axis=1)
test = test.drop('Original_Quote_Date',axis=1)
# Extract year,month,weekday from 'Date'
train['Year'] = train['Date'].apply(lambda x:x.year)
train['Month'] = train['Date'].apply(lambda x:x.year)
train['Weekday'] = train['Date'].apply(lambda x:x.weekday())

test['Year'] = test['Date'].apply(lambda x:x.year)
test['Month'] = test['Date'].apply(lambda x:x.year)
test['Weekday'] = test['Date'].apply(lambda x:x.weekday())
# Drop 'Date' feature
train = train.drop('Date',axis=1)
test = test.drop('Date',axis=1)

# # One-Hot Encoding for Categorical features
# train = pd.get_dummies(train)
# test = pd.get_dummies(test)
train.isnull().sum().sort_values(ascending = False)
test.isnull().sum().sort_values(ascending = False).head(12)
train[train.columns[train.isnull().any()]].head(10)
test[test.columns[test.isnull().any()]].head(10)
# LabelEncode categorical features
for c in train.columns:
    if train[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train[c].values) + list(test[c].values))
        train[c] = lbl.transform(list(train[c].values))
        test[c] = lbl.transform(list(test[c].values))
        
# Fill median in numerical features
train = train.fillna(train.median())
test = test.fillna(test.median())
train.isnull().sum().sort_values(ascending = False)
test.isnull().sum().sort_values(ascending = False).head(3)
%env JOBLIB_TEMP_FOLDER=/tmp
if __name__ == '__main__':
    xgb_model = xgb.XGBClassifier(
    )
    parameters = {
              'silent': [1],
              'learning_rate': [0.1],
              'min_child_weight': [5],
              'max_depth': [6],
              'subsample': [0.8],
#               'subsample': [0.8, 1.0],
#               'colsample_bytree': [0.8, 1.0],
              'colsample_bytree': [0.8],
              'objective': ['binary:logistic'],
              'n_estimators': [5],
              'seed': [1337]
    }
    clf = GridSearchCV(xgb_model, param_grid=parameters, n_jobs=5,
                  cv=StratifiedKFold(n_splits = 5, shuffle=True),
                  scoring='roc_auc',
                  verbose=2,
                  refit=True)

    clf.fit(train[list(train.columns[1:])], train['QuoteConversion_Flag'])
    best_parameters, score, _ = max(clf.grid_scores_, key=lambda x:x[1])
    print('Raw AUC score:', score)
    for param_name in sorted(best_parameters.keys()):
        print("%s: %r" % (param_name, best_parameters[param_name]))
    
    test_probs = clf.predict_proba(test[list(train.columns[1:])])[:,1]
# parameters = {'nthread': [4],
#               'silent': [1],
#               'learning_rate': [0.05],
#               'min_child_weight': [11],
#               'max_depth': [5],
#               'subsample': [0.8],
#               'colsample_bytree': [0.7],
#               'objective': ['binary:logistic'],
#               'n_estimators': [5],
#               'seed': [1337]}
# clf = GridSearchCV(xgb_model, param_grid=parameters, n_jobs=5,
#                   cv=StratifiedKFold(n_splits = 5, shuffle=True),
#                   scoring='roc_auc',
#                   verbose=2,
#                   refit=True)

# clf.fit(train[list(train.columns[1:])].head(10000), train['QuoteConversion_Flag'].head(10000))
# best_parameters, score, _ = max(clf.grid_scores_, key=lambda x:x[1])
# print('Raw AUC score:', score)
# for param_name in sorted(best_parameters.keys()):
#     print("%s: %r" % (param_name, best_parameters[param_name]))
    
# test_probs = clf.predict_proba(test[list(train.columns[1:])])[:,1]



