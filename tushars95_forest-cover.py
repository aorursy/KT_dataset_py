import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import os
print(os.listdir("../input"))
#Read data for analysis
train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/train.csv')
sampleSubmission=pd.read_csv('../input/sampleSubmission.csv')
X=train.loc[:,'Elevation':'Soil_Type40']
y=train['Cover_Type']
X_t=test.loc[:,'Elevation':'Soil_Type40']
y_t=test['Cover_Type']
print(X_t.shape)
#Splitting the data into  train and test
X_train, X_test1, y_train, y_test1 = train_test_split(X,y, test_size=0.3, random_state=100)
print(X_train.shape)
print(X_test1.shape)
print(y_train.shape)
print(y_test1.shape)
model_rf = RandomForestClassifier(random_state=100)
model_rf.fit(X_train, y_train)
test_pred_rf = model_rf.predict(X_test1)
df_pred_rf = pd.DataFrame({'actual': y_test1,
                        'predicted': test_pred_rf})
df_pred_rf.head(5)
Accuracy=model_rf.score(X_test1,y_test1)
print('Random Forest Accuracy:',Accuracy)
test_pred = model_rf.predict(X_t)
df_test_pred = pd.DataFrame(test_pred, columns=['Cover_Type'])
df_test_pred['Id'] = X_t.index + 1
df_test_pred[['Id', 'Cover_Type']].to_csv('submission.csv', index=False)
pd.read_csv('submission.csv').head()
