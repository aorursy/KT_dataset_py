# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping,ModelCheckpoint

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/titanic/train.csv')
train.info()
age_mean = train['Age'].mean()
train['Age'].fillna(age_mean, inplace = True)
train['Embarked'].fillna('S', inplace = True)
train.info()
bins = pd.IntervalIndex.from_tuples([(0, 10), (10, 18), (18, 25), (25, 40), (40, 60), (60, 80)])
train['Age'] = pd.cut(train['Age'], bins)

train.head()
bins = pd.IntervalIndex.from_tuples([(0, 7.91), (7.91, 14.454), (14.454, 31), (31, 512.329)])
train['Fare'] = pd.cut(train['Fare'], bins)

train
train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
    

train['IsAlone'] = 0
train.loc[train['FamilySize'] == 1, 'IsAlone'] = 1

train
df = train[['Sex','Fare','IsAlone','FamilySize','Pclass','Age','Embarked']]
df
X = pd.get_dummies(df)
y = pd.get_dummies(train[['Survived']])

X_train = X[:650]
X_val = X[240:]
y_train = y[:650]
y_val = y[240:]

X
model = Sequential()
model.add(Dense(18, input_dim=18, activation='relu'))
model.add(Dense(72, activation='relu'))
model.add(Dense(54, activation='relu'))
model.add(Dense(36, activation='relu'))
model.add(Dense(18, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(
    optimizer=SGD(learning_rate=0.01),  # Optimizer
    # Loss function to minimize
    loss=keras.losses.BinaryCrossentropy(),
    # List of metrics to monitor
    metrics=['accuracy']
)
callback1 = EarlyStopping(monitor='val_loss', 
                         patience=50,
                         restore_best_weights=True
                        )


checkpoint_filepath = '/tmp/checkpoint'

model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

history = model.fit(
    X_train,
    y_train,
    batch_size=1,
    epochs=1000,
    callbacks=[callback1,model_checkpoint_callback],
    validation_data= (X_val, y_val)
    )
        # We pass some validation for
        # monitoring validation loss and metrics
        # at the end of each epoch
# The model weights (that are considered the best) are loaded into the model.
model.load_weights(checkpoint_filepath)
from sklearn.metrics import confusion_matrix

testingNN = model.predict(
    X_val,
    steps=None,
    callbacks=None,
    max_queue_size=10,
    workers=1,
    use_multiprocessing=False,
)

prediction = []
for res in testingNN:
    if res[0]>0.435:
        prediction.append(1)
    else:
        prediction.append(0)

y_true = np.array(y_val)
pd.DataFrame(confusion_matrix(y_true, prediction))
test = pd.read_csv('/kaggle/input/titanic/test.csv')
test['Age'].fillna(age_mean, inplace = True)
test.info()
bins = pd.IntervalIndex.from_tuples([(0, 7.91), (7.91, 14.454), (14.454, 31), (31, 512.329)])
test['Fare'] = pd.cut(test['Fare'],bins)

bins = pd.IntervalIndex.from_tuples([(0, 10), (10, 18), (18, 25), (25, 40), (40, 60), (60, 80)])
test['Age'] = pd.cut(test['Age'],bins)

test['FamilySize'] = test['SibSp'] + test['Parch'] + 1
    

test['IsAlone'] = 0
test.loc[test['FamilySize'] == 1, 'IsAlone'] = 1



df2 = test[['Sex','Fare','IsAlone','FamilySize','Pclass','Age','Embarked']]
X_test = pd.get_dummies(df2)
X_test
Results = model.predict(
    X_test,
    batch_size=1,
    steps=None,
    callbacks=None,
    max_queue_size=10,
    workers=1,
    use_multiprocessing=False,
)
predicted = []
for res in Results:
    if res[0]>=0.435:
        predicted.append(1)
    else:
        predicted.append(0)
final_df = pd.DataFrame(test["PassengerId"]) 
final_df["Survived"] = predicted
final_df
final_df.to_csv('NN.csv',index=False)
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
dicts = {}


train_dtc = pd.read_csv('/kaggle/input/titanic/train.csv')
train_dtc = train_dtc.drop(['PassengerId','Name','Cabin','Ticket','Fare'],axis=1)

train_dtc = pd.DataFrame(train_dtc.dropna())
verif_dtc = pd.DataFrame(train_dtc['Survived'])
train_dtc = (train_dtc.drop(['Survived'],axis=1))

train_dtc = pd.get_dummies(train_dtc)

print(verif_dtc.info(),train_dtc.info())
from sklearn.ensemble import RandomForestClassifier
np.random.seed(42)

rfc=RandomForestClassifier(n_estimators = 100, 
                           max_features='auto', 
                           criterion='entropy',
                           max_depth=10)

rfc.fit(X_train,y_train.values.ravel())
rfc_preds= rfc.predict(X_val)

from sklearn.metrics import accuracy_score

accuracy_score(y_val.values.ravel(),rfc_preds)
res2 = pd.DataFrame(test["PassengerId"]) 
rfc_preds= rfc.predict(X_test)
result2 = pd.DataFrame(rfc_preds)
res2['Survived'] = result2
res2
res2.to_csv('RFC.csv',index=False)
train_dtc = pd.read_csv('/kaggle/input/titanic/train.csv')
train_dtc = train_dtc.drop(['PassengerId','Name','Cabin','Ticket','Fare'],axis=1)

train_dtc = pd.DataFrame(train_dtc.dropna())
verif_dtc = pd.DataFrame(train_dtc['Survived'])
train_dtc = (train_dtc.drop(['Survived'],axis=1))


print(verif_dtc.info(),train_dtc.info())
test_dtc = pd.read_csv('/kaggle/input/titanic/test.csv')

test_dtc = test_dtc.drop(['PassengerId','Name','Cabin','Ticket','Fare'],axis=1)
test_dtc['Age'].fillna(age_mean, inplace = True)

print(test_dtc.info())
train_dtc = pd.get_dummies(train_dtc)
test_dtc = pd.get_dummies(test_dtc)
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(random_state=0)
dtc.fit(train_dtc,verif_dtc)
result_dtc = dtc.predict(test_dtc)
res3 = pd.DataFrame(test["PassengerId"]) 
result3 = pd.DataFrame(result_dtc)
res3['Survived'] = result3
res3
res3.to_csv('DTC.csv',index=False)
from sklearn.neighbors import KNeighborsClassifier

train_knc = pd.read_csv('/kaggle/input/titanic/train.csv')
train_knc = train_knc.drop(['PassengerId','Name','Cabin','Ticket','Fare'],axis=1)

train_knc = pd.DataFrame(train_knc.dropna())
verif_knc = pd.DataFrame(train_knc['Survived'])
train_knc = (train_knc.drop(['Survived'],axis=1))
train_knc = pd.get_dummies(train_knc)

print(verif_knc.info(),train_knc.info())
test_dtc = pd.read_csv('/kaggle/input/titanic/test.csv')

test_dtc = test_dtc.drop(['PassengerId','Name','Cabin','Ticket','Fare'],axis=1)
test_dtc['Age'].fillna(age_mean, inplace = True)
test_dtc = pd.get_dummies(test_dtc)
print(test_dtc.info())
knc = KNeighborsClassifier(n_neighbors=7)
knc.fit(train_knc,verif_knc.values.ravel())
res_knc =  knc.predict(test_dtc)
res4 = pd.DataFrame(test["PassengerId"]) 
result4 = pd.DataFrame(res_knc)
res4['Survived'] = result4
res4
res4.to_csv('KNC.csv',index=False)
sum_res = pd.DataFrame([predicted,
rfc_preds,
result_dtc,
res_knc])
a = sum_res.sum()
res = pd.DataFrame(a, columns=['data'])
res['Survived'] = (res.data>=2).astype(int)
res = res.drop(['data'], axis=1)

final = pd.DataFrame(test["PassengerId"])
final['Survived'] = res
final
final.to_csv('Total.csv',index=False)
