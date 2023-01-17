

import numpy as np # linear algebra
import pandas as pd # data processing

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

train.describe()

print (train.shape)
print (test.shape)
print ("\n train:\n ",train.head(),"\n")
print ("test: \n",test.head())
print ("unique place :",train.place_id.unique().shape)
#print ("place_id   value_count")
#print(train.place_id.value_counts())



import matplotlib.pyplot as plt

groups = train.groupby('place_id')
count =0
# Plot
fig, ax = plt.subplots()

for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, label=name)
    count =count+1
    if count>10:
        break
ax.legend()

plt.show()
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.svm import OneClassSVM
y_train = train['place_id'].values
X_train = train.drop(['row_id','place_id'], axis=1).values
X_fit, X_eval, y_fit, y_eval= train_test_split(X_train, y_train, test_size=0.3)

id_test = test['row_id']
X_test = test.drop(['row_id'], axis=1).values
clf2=RandomForestClassifier()
clf2.fit(X_train, y_train)
print('Overall AUC:', roc_auc_score(y_train, clf2.predict_proba(X_train)[:,1]))

y_pred2= clf2.predict_proba(X_test)[:,1]
submission = pd.DataFrame({"row_id":id_test, "place_id":y_pred2})
submission.to_csv("submission.csv", index=False)

