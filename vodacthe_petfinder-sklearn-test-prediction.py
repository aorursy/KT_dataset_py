# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
breed_labels = pd.read_csv("../input/breed_labels.csv")
color_labels = pd.read_csv("../input/color_labels.csv")
state_labels = pd.read_csv("../input/state_labels.csv")
train = pd.read_csv('../input/train/train.csv') 
test = pd.read_csv('../input/test/test.csv')
train['dataset_type'] = 'train'
test['dataset_type'] = 'test'
train=train.drop("Description",axis=1) #Drop description column (axis 1 = Drop labels from column )
test=test.drop("Description",axis=1)
all_data = pd.concat([train, test])

all_data.head()
all_data.info()
train.info()
df = train['AdoptionSpeed'].value_counts()

ax = df.plot("bar",color="green",title="Adoption speed classes counts");
ax.set_xticklabels(["No adoption","31-90 days","8-30 days","1-7 days","0 days"])

train['Age'].value_counts().head(10)
print('Dogs and cat in the dataset')
af = train['Type'].value_counts()
print(af)
#-------------------
query = train.query('AdoptionSpeed == 4')
df = query['Type'].value_counts()
#-----------------
print('\nDogs and cat not adopted')
print(df)
print('\nPercentage of adoption of Dogs')
dogsP = (af[1] - df[1]) / af[1] * 100
print(dogsP)
print('\nPercentage of adoption of Cats')
dogsP = (af[2] - df[2]) / af[2] * 100
print(dogsP)

print('Fees in the dataset')
af = train['Fee'].value_counts().head(10)
print(af)
#-------------------
query = train.query('AdoptionSpeed == 4')
df = query['Fee'].value_counts().head(10)
#-----------------
print('\nFees of the pet that was not adopted')
print(df)
a = all_data
a['Type'] = a['Type'].apply(lambda x: 'Dog' if x == 1 else 'Cat')
sns.countplot(x='dataset_type', data=a, hue='Type')
plt.title('Number of cats and dogs in train and test data')
query = train.query('AdoptionSpeed == 0')
df = query['Age'].value_counts()
ax = df.plot("bar",color="green",title="Same day adopted by age",figsize=(16,3));
#ax.set_xticklabels(["Dogs","Cats"])
query = train.query('AdoptionSpeed == 1')
df = query['Age'].value_counts()
ax = df.plot("bar",color="green",title="First week adopted by age",figsize=(16,3));
#ax.set_xticklabels(["Dogs","Cats"])
query = train.query('AdoptionSpeed == 2')
df = query['Age'].value_counts()
ax = df.plot("bar",color="green",title="First month adopted by age",figsize=(16,3));
#ax.set_xticklabels(["Dogs","Cats"])
query = train.query('AdoptionSpeed == 0')
df = query['Breed1'].value_counts()
ax = df.plot("bar",color="green",title="Same day adopted by breed",figsize=(16,3));
ax.set_xticklabels(breed_labels['BreedName'],zorder=breed_labels['BreedID'])
train['Age'].value_counts().head(10)
train['Name'].value_counts().head(10)
sns.countplot(x='AdoptionSpeed',data=train,hue='Gender')
sns.countplot(x='AdoptionSpeed',data=train,hue='MaturitySize')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model
%matplotlib inline

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

train = pd.read_csv('../input/train/train.csv') 
test = pd.read_csv('../input/test/test.csv')
sub = pd.read_csv('../input/test/sample_submission.csv')
train['dataset_type'] = 'train'
test['dataset_type'] = 'test'

train=train.drop("Description",axis=1) #Drop description column (axis 1 = Drop labels from column )
test=test.drop("Description",axis=1)
all_data = pd.concat([train, test])
all_data_t = all_data
all_data_t = all_data_t.drop("Name",axis=1)
all_data_t = all_data_t.drop("PetID",axis=1)
all_data_t = all_data_t.drop("RescuerID",axis=1)
all_data_t = all_data_t.drop("dataset_type",axis=1)
all_data_t = all_data_t.drop("PhotoAmt",axis=1)
all_data_t = all_data_t.drop("VideoAmt",axis=1)
all_data_t = all_data_t.drop("Quantity",axis=1)
all_data_t = all_data_t.drop("AdoptionSpeed",axis=1)


all_data_t.head()
from sklearn.preprocessing import OneHotEncoder
hot = OneHotEncoder(categories='auto',sparse=False)

all_hot=hot.fit_transform(all_data_t)
all_hot
all_hot = np.append(all_hot,all_data['AdoptionSpeed'].values.reshape(18941,1),axis=1)
train_x=all_hot[:14993,:-1]
train_y=all_hot[:14993,-1]
test_x=all_hot[14993:,:-1]
test_y=all_hot[14993:,-1]
from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y=train_test_split(train_x,train_y,test_size=0.2)
train_x.shape
from sklearn.linear_model import LogisticRegression
logReg = LogisticRegression(penalty="l1",C=0.0001,intercept_scaling=0.0001,random_state=0,solver="liblinear")
logReg.fit(train_x, train_y)
predict = logReg.predict(test_x)
logReg.score(train_x, train_y)
plot_learning_curve(logReg, "Learning Curves Logistic Regression", train_x, train_y)
plt.show()
from xgboost import XGBClassifier
xgb = XGBClassifier(learning_rate=0.0001,booster='dart',max_delta_step=1,subsample=0.05)
xgb.fit(train_x, train_y)
xPredict = xgb.predict(test_x)
xgb.score(train_x,train_y)
plot_learning_curve(xgb, "Learning Curves XGB", train_x, train_y)
plt.show()
from sklearn import tree
decTree = tree.DecisionTreeClassifier()
decTree.fit(train_x,train_y)
treePredict = decTree.predict(test_x)
decTree.score(train_x,train_y)
plot_learning_curve(decTree, "Learning Curves Decision Tree", train_x, train_y)
plt.show()
from keras.models import Sequential
from keras.layers import Conv1D,Dense,MaxPool1D,Flatten,Dropout
import keras
from keras import callbacks
train_y_lb = hot.fit_transform(list(train_y.reshape(-1, 1)))
test_y_lb = hot.fit_transform(list(test_y.reshape(-1, 1)))
train_y_lb.shape

model = Sequential()
model.add(Dense(1,activation='relu'))
model.add(Dropout(0.05))
model.add(Dense(32,activation='relu'))
model.add(Dropout(0.05))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.05))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.05))
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.05))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.05))
model.add(Dense(5,activation='sigmoid'))
#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.compile(optimizer='adam',
              loss='logcosh',
              metrics=['accuracy'])
model.fit(train_x, train_y_lb, epochs=10,validation_data=(test_x,test_y_lb),verbose=1, batch_size=32)

score = model.evaluate(test_x, test_y_lb, batch_size=32)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
deepPredicts = model.predict(test_x,verbose=1)

predict = np.zeros((len(deepPredicts),1))
predict = np.argmax(deepPredicts,axis = 1)
    
predict
sub=pd.read_csv('../input/test/sample_submission.csv')
sub.head()
for i,val in enumerate(predict):
    sub.at[i,'AdoptionSpeed'] = val
sub.AdoptionSpeed = sub.AdoptionSpeed.astype(int)
sub.head()
len(sub)
sub.to_csv('submission.csv', index=False)
sub.shape
