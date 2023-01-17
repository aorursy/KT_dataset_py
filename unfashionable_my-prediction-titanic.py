# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv",index_col = 0)
test = pd.read_csv("../input/test.csv",index_col = 0)
train.head()
train = train.drop(['Name','Ticket','Cabin'],axis=1)
test = test.drop(['Name','Ticket','Cabin'],axis=1)
train.head()
sex_map = {}
sex_map['male'] = 0
sex_map['female'] = 1
train.Sex = train.Sex.map(sex_map)


train.head()
test.Sex = test.Sex.map(sex_map)


train.Embarked.unique()
em_map = {}
em_map['S'] = 0
em_map['C'] = 1
em_map['Q'] = 2
train.Embarked = train.Embarked.map(em_map)
test.Embarked = test.Embarked.map(em_map)
train.head()
for c in train.columns:
    print (c,train[c].unique())
train['Sex']
Y = train.Survived.tolist()
train = train.drop(['Survived'],axis=1)
for c in train.columns:
    train[c] = train[c].fillna(train[c].median())
    test[c] = test[c].fillna(train[c].median())    
train.head()
from sklearn.manifold import TSNE
data = pd.concat([train,test])
data.head()
TSNE_features = TSNE(n_components = 3).fit_transform(data)
new_features = pd.DataFrame(TSNE_features)
new_features.head()
train_rows = train.shape[0]
test_rows = test.shape[0]
train['TSNE1'] = new_features.iloc[0:(train_rows-1),0]
train['TSNE2'] = new_features.iloc[0:(train_rows-1),1]
train['TSNE3'] = new_features.iloc[0:(train_rows-1),2]

print (train.shape)
print (test.shape)
new_features.shape
test['TSNE1'] = new_features.iloc[train_rows:(train_rows+test_rows-1),0]
test['TSNE2'] = new_features.iloc[train_rows:(train_rows+test_rows-1),1]
test['TSNE3'] = new_features.iloc[train_rows:(train_rows+test_rows-1),2]
train.head()
from mlxtend.classifier import StackingCVClassifier as SCVC
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.linear_model import RidgeClassifier as RC
clf1 = kNN()
clf2 = SVC(probability=True)
clf3 = RFC()
meta_clf = RC()
stacker = SCVC(classifiers = [clf1,clf2,clf3,clf1],meta_classifier = meta_clf,use_probas=True, use_features_in_secondary=True)
for c in train.columns:
    train[c] = train[c].fillna(train[c].median())
    test[c] = test[c].fillna(train[c].median())    
stacker.fit(train.values,np.array(Y))
my_prediction = stacker.predict(test.values)
# PassengerId,Survived
submission = pd.DataFrame()
submission['PassengerId'] = test.index.tolist()
submission['Survived'] = my_prediction
submission.to_csv("submission.csv",index=False)
