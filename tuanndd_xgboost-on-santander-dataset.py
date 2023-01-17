import numpy as np

import pandas as pd



from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier 

from sklearn.cross_validation import cross_val_score, train_test_split



from xgboost import XGBClassifier



import matplotlib.pyplot as plt



import time



%pylab inline
# Load data

df_train=pd.read_csv('../input/train.csv')

df_test=pd.read_csv('../input/test.csv')
# Data summary

print(df_train.shape)



print(df_train.TARGET.value_counts(normalize=True))

# Imbalanced classes!
desc=df_train.describe()

desc.loc['unique']=[len(df_train[i].unique()) for i in df_train.columns]

desc=desc.T



desc.to_csv('data_summary.csv')

# Some features have unique col = 0, some features have max col = 9999999999 (maybe outlier values)

desc.head()
# Preprocess data
# Remove duplicated columns

remove=[]



cols=df_train.columns

for i in range(len(cols)-1):

    for j in range(i+1, len(cols)):

        if np.array_equal(df_train[cols[i]].values, df_train[cols[j]].values):

            remove.append(cols[j])



df_train.drop(remove, axis=1, inplace=True)

df_test.drop(remove, axis=1, inplace=True)
# Remove constant columns (std=0)

remove=[]



for col in df_train.columns:

    if df_train[col].std()==0:

        remove.append(col)



df_train.drop(remove, axis=1, inplace=True)

df_test.drop(remove, axis=1, inplace=True)
# Split data in train and test

X=df_train.drop(['ID','TARGET'], axis=1)

y=df_train.TARGET.values



test_id=df_test.ID

test=df_test.drop(['ID'], axis=1)



X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.2, random_state=0)



print(X_train.shape)
# Model selection

def score_model(clf):

    print ("model: {} ...".format(clf.__class__.__name__))

    start = time.time()

    scores = cross_val_score(clf, X_train, y_train, scoring='roc_auc', cv=3) 

    end = time.time()

    print("time: {:.3f}s".format(end - start))

    print("roc_auc: {:.3f}\n".format(scores.mean()))
score_model(DecisionTreeClassifier())

score_model(GaussianNB())

score_model(LogisticRegression())

score_model(RandomForestClassifier())

score_model(XGBClassifier())
# Feature selection



clf=XGBClassifier()



clf.fit(X_train,y_train)



importances=clf.booster().get_fscore()



df_importance=pd.Series(list(importances.values()), index=list(importances.keys()))

df_importance.sort_values(inplace=True, ascending=False)
# Top importance features

ax=df_importance.head(100).plot(kind='barh', figsize=(10,20))

ax.invert_yaxis()
df_importance
imp_cols=df_importance[df_importance>1].index.tolist()

print(imp_cols,"\n", len(imp_cols))
# Tuning parameters of xgboost:

# Reference: https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/



# Result:

# clf=XGBClassifier(n_estimators=1000, learning_rate=0.01, max_depth=4, min_child_weight=4, gamma=0, colsample_bytree=0.7, subsample=0.6, reg_alpha=5e-05, objective='binary:logistic', scale_pos_weight=1, seed=0)
# Predict test
# Train model

eval_metrics = ['auc']

eval_sets = [(X_train, y_train), (X_test, y_test)]



clf=XGBClassifier(n_estimators=1000, learning_rate=0.01, max_depth=4, min_child_weight=4, gamma=0, colsample_bytree=0.7, subsample=0.6, reg_alpha=5e-05, objective='binary:logistic', scale_pos_weight=1, seed=0)



clf.fit(X_train, y_train, eval_metric=eval_metrics, eval_set=eval_sets)
# Create submission

y_pred=clf.predict_proba(test)



df_submit=pd.DataFrame({'ID': test_id, 'TARGET': y_pred[:,-1]})

df_submit.to_csv('submission.csv', index=False)



df_submit.head()
# Score on LB = 0.825860