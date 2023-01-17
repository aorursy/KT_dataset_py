# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv('../input/titanic/train.csv')

train_data.head()
test_data = pd.read_csv('../input/titanic/test.csv')

test_data.head()
women = train_data.loc[train_data.Sex == 'female']['Survived']

rate_women = sum(women)/len(women)



print('% of women who survived:', rate_women)

print('survived rate of women is {}%'.format(rate_women))

men = train_data[train_data['Sex'] == 'male']['Survived']

rate_men = sum(men)/len(men)



print('% of men who survived:', rate_men)

train_data.info()
## null check

train_data.isnull().any()
train_data['Cabin'].unique()
train_data['Embarked'].value_counts()
from sklearn.impute import SimpleImputer



train_data = train_data.dropna(subset=['Embarked'])



train_data['Age'] = train_data['Age'].apply(lambda x: round(x, 0))

test_data['Age'] = test_data['Age'].apply(lambda x: round(x, 0))



imp = SimpleImputer(strategy='most_frequent')

train_imp = imp.fit_transform(train_data['Age'].values.reshape(-1, 1))

test_imp = imp.transform(test_data['Age'].values.reshape(-1, 1))



train_data['Age'] = train_imp

test_data['Age'] = test_imp



train_data['Cabin'] = train_data['Cabin'].fillna('nan')

test_data['Cabin'] = test_data['Cabin'].fillna('nan')



train_data['Cabin_cnt'] = train_data['Cabin'].apply(lambda x: len(x.split(' ')) if x != 'nan' else 0)

test_data['Cabin_cnt'] = test_data['Cabin'].apply(lambda x: len(x.split(' ')) if x != 'nan' else 0)



print(train_data.isnull().any())
print(train_data['SibSp'].unique())

print(test_data['SibSp'].unique())
import pickle

from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder



y = train_data['Survived'].values



features = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Cabin_cnt', 'Embarked', 'Age']



X = train_data[features]

X_test = test_data[features]



print(X.shape)

print(X.columns)

print(X_test.shape)

print(X_test.columns)



X = pd.get_dummies(train_data[features])

X_test = pd.get_dummies(test_data[features])



print(X)

print(y)

# save the datas

np.savez_compressed('np_savez_comp', X=X, y=y)
!ls
# load from npz

datas = np.load('np_savez_comp.npz', allow_pickle=True)

print(datas.files)
X = datas['X']

y = datas['y']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print(X_train)

print(y_train)
score_lists = []
from sklearn.svm import SVC

svm = SVC(kernel='linear', C=1.0, random_state=0)

svm.fit(X_train, y_train)



with open('Support Vector Machine.pickle', mode='wb') as fp:

    pickle.dump(svm, fp)

score = svm.score(X_test, y_test)

print('score: {}' .format(score))
# カーネルで曲線で判定した場合(rbf)

svm2 = SVC(kernel='rbf', C=1.0, random_state=0)

svm2.fit(X_train, y_train)
score = svm2.score(X_test, y_test)

print('score: {}' .format(score))
# gammaを補正

svm3 = SVC(kernel='rbf', C=1.0, gamma=0.10, random_state=0)

svm3.fit(X_train, y_train)

score = svm3.score(X_test, y_test)

print('score: {}' .format(score))
# gammaを補正

svm3 = SVC(kernel='rbf', C=1.0, gamma=100.0, random_state=0)

svm3.fit(X_train, y_train)

score = svm3.score(X_test, y_test)

print('score: {}' .format(score))
from sklearn.linear_model import LogisticRegression

logistic_regression = LogisticRegression(penalty='l2', C=100, random_state=0, max_iter=1000)

logistic_regression.fit(X_train, y_train)



with open('Logistic Regression.pickle', mode='wb') as fp:

    pickle.dump(logistic_regression, fp)

score = logistic_regression.score(X_test, y_test)

print('score: {}' .format(score))
from sklearn.tree import DecisionTreeClassifier

desicion_tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)

desicion_tree.fit(X_train, y_train)



with open('Decision Tree.pickle', mode='wb') as fp:

    pickle.dump(desicion_tree, fp)

score = desicion_tree.score(X_test, y_test)

print('score: {}' .format(score))
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')

knn.fit(X_train, y_train)



with open('Nearest Neighbors.pickle', mode='wb') as fp:

    pickle.dump(knn, fp)

score = knn.score(X_test, y_test)

print('score: {}' .format(score))
from sklearn.naive_bayes import GaussianNB



gnb = GaussianNB()

gnb.fit(X_train, y_train)



with open('Naive Bayes.pickle', mode='wb') as fp:

    pickle.dump(gnb, fp)

score = gnb.score(X_test, y_test)

print('score: {}' .format(score))
from sklearn.ensemble import RandomForestClassifier



random_forest = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)

random_forest.fit(X_train, y_train)



with open('Random Forest.pickle', mode='wb') as fp:

    pickle.dump(random_forest, fp)

score = random_forest.score(X_test, y_test)

print('score: {}' .format(score))
from sklearn.ensemble import AdaBoostClassifier



ada_boost = AdaBoostClassifier(base_estimator=None, n_estimators=50, learning_rate=1.0, algorithm='SAMME.R', random_state=None)

ada_boost.fit(X_train, y_train)



with open('Ada Boost.pickle', mode='wb') as fp:

    pickle.dump(ada_boost, fp)

score = ada_boost.score(X_test, y_test)

print('score: {}' .format(score))
names = ["Support Vector Machine", "Logistic Regression", "Nearest Neighbors",

         "Decision Tree","Random Forest", "Naive Bayes", "Ada Boost"]



# アルゴリズムを順に実行

result = []



for name in names:

    with open(name + '.pickle', 'rb') as fp:

        clf = pickle.load(fp)

    

#     clf.fit(X_train, y_train)

    score1 = clf.score(X_train, y_train)

    score2 = clf.score(X_test, y_test)

    result.append([score1, score2])

    

    print(name)



df_result = pd.DataFrame(result, columns=['train', 'test'], index = names)

df_result

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score



kfold = KFold(n_splits=5, random_state=0, shuffle=True)



# svm

result = cross_val_score(svm, X, y, cv=kfold, scoring='accuracy')

print('svm:{}'.format(result.mean()))



# logistic_regression

result = cross_val_score(logistic_regression, X, y, cv=kfold, scoring='accuracy')

print('logistic regression:{}'.format(result.mean()))



# desicion_tree

result = cross_val_score(desicion_tree, X, y, cv=kfold, scoring='accuracy')

print('desicion tree:{}'.format(result.mean()))



# knn

result = cross_val_score(knn, X, y, cv=kfold, scoring='accuracy')

print('knn:{}'.format(result.mean()))



# random_forest

result = cross_val_score(random_forest, X, y, cv=kfold, scoring='accuracy')

print('random forest:{}'.format(result.mean()))

from sklearn.model_selection import GridSearchCV



params = {'max_depth': list(range(2, 10)), 

          'min_samples_leaf': list(range(1, 12, 2))}



grid_search = GridSearchCV(random_forest, params, cv=5, return_train_score=True)

# grid_search = GridSearchCV(random_forest, params, cv=5, return_train_score=True, verbose=3)

grid_search.fit(X, y)
print('best score: {:0.3f}'.format(grid_search.score(X, y)))

print('best params: {}'.format(grid_search.best_params_))

print('best val score:  {:0.3f}'.format(grid_search.best_score_))
import seaborn as sns

%matplotlib inline



cv_result = pd.DataFrame(grid_search.cv_results_)

cv_result = cv_result[['param_max_depth', 'param_min_samples_leaf', 'mean_test_score']]

cv_result_pivot = cv_result.pivot_table('mean_test_score', 'param_max_depth', 'param_min_samples_leaf')



heat_map = sns.heatmap(cv_result_pivot, cmap='Greys', annot=True);
from xgboost import XGBClassifier



xgb = XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=100)

xgb.fit(X_train, y_train)
score = xgb.score(X_test, y_test)

print('score: {}' .format(score))
from lightgbm import LGBMClassifier



lgb = LGBMClassifier(max_depth=3, learning_rate=0.1, n_estimators=100)

lgb.fit(X_train, y_train)
score = lgb.score(X_test, y_test)

print('score: {}' .format(score))
params = {'max_depth': list(range(2, 10)),

          'eta': [0.01, 0.1, 1.0],

          'gamma': [0, 0.1],

          'min_child_weight': [1, 2],

          'nthread': [2, 4],

          'n_estimators': list(range(50, 200, 50))}



xgb = XGBClassifier()



reg_cv = GridSearchCV(xgb, params, cv=5, return_train_score=True)

# reg_cv = GridSearchCV(xgb, params, cv=5, return_train_score=True, verbose=3)

reg_cv.fit(X_train, y_train)

with open('XGBoost.pickle', mode='wb') as fp:

    xgb = XGBClassifier(**reg_cv.best_params_)

    xgb.fit(X_train, y_train)

    pickle.dump(xgb, fp)

print('best score: {:0.3f}'.format(reg_cv.score(X_train, y_train)))

print('best params: {}'.format(reg_cv.best_params_))

print('best val score:  {:0.3f}'.format(reg_cv.best_score_))
xgb = reg_cv.best_estimator_

score = xgb.score(X_test, y_test)

print('score: {}' .format(score))
lgb = LGBMClassifier()



reg_cv = GridSearchCV(lgb, params, cv=5, return_train_score=True)

# reg_cv = GridSearchCV(lgb, params, cv=5, return_train_score=True, verbose=3)

reg_cv.fit(X_train, y_train)

with open('LightGBM.pickle', mode='wb') as fp:

    lgb = LGBMClassifier(**reg_cv.best_params_)

    lgb.fit(X_train, y_train)

    pickle.dump(lgb, fp)

print('best score: {:0.3f}'.format(reg_cv.score(X_train, y_train)))

print('best params: {}'.format(reg_cv.best_params_))

print('best val score:  {:0.3f}'.format(reg_cv.best_score_))
lgb = reg_cv.best_estimator_

score = lgb.score(X_test, y_test)

print('score: {}' .format(score))
from sklearn.ensemble import AdaBoostClassifier



names = ["Support Vector Machine", "Logistic Regression", "Nearest Neighbors",

         "Decision Tree","Random Forest", "Ada Boost", 

         "Naive Bayes", 'XGBoost', 'LightGBM']



# アルゴリズムを順に実行

result = []



for name in names:

    with open(name + '.pickle', 'rb') as fp:

        clf = pickle.load(fp)



#     clf.fit(X_train, y_train)

    score1 = clf.score(X_train, y_train)

    score2 = clf.score(X_test, y_test)

    result.append([score1, score2])

    print(name)



df_result = pd.DataFrame(result, columns=['train', 'test'], index = names)

df_result

batch_size = 64

n_epochs = 100
X_train.shape
import tensorflow as tf

from tensorflow import keras



model = keras.Sequential([

    keras.layers.Dense(512, activation='relu', input_shape=[X_train.shape[1]]),

    keras.layers.Dropout(0.2),

    keras.layers.Dense(512, activation='relu'),

    keras.layers.Dropout(0.2),

    keras.layers.Dense(2, activation='softmax'),

])
model.compile(optimizer='adam', 

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])
model.fit(X_train, y_train, epochs=n_epochs, batch_size=batch_size)
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)



print('\nTest accuracy:', test_acc)
import torch

from torch.autograd import Variable

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset



train_X = torch.tensor(X_train, dtype=torch.float32)

train_y = torch.tensor(y_train)

test_X = torch.tensor(X_test, dtype=torch.float32)

test_y = torch.tensor(y_test)
train = TensorDataset(train_X, train_y)

print(train[0])



train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
class Net(nn.Module):

    def __init__(self, col_num):

        super(Net, self).__init__()

        self.fc1 = nn.Linear(col_num, 512)

        self.fc2 = nn.Linear(512, 512)

        self.fc3 = nn.Linear(512, 2)

        self.dropout = nn.Dropout(0.2)

    

    def forward(self, x):

        x = F.relu(self.fc1(x)) # ReLU: max(x, 0)

        x = self.dropout(x)

        x = F.relu(self.fc2(x))

        x = self.dropout(x)

        x = self.fc3(x)

        return x

    

net = Net(X_train.shape[1])
# 損失関数

criterion = nn.CrossEntropyLoss() # 他クラス分類:ソフトマックス交差エントロピー誤差

#criterion = nn.MSELoss() # 回帰:平均二乗誤差

#criterion = nn.L1Loss() # 回帰:平均絶対値誤差

# criterion = nn.BCELoss() # 二値分類:バイナリ交差エントロピー

# criterion = nn.BCEWithLogitsLoss() # 二値分類:ロジット・バイナリ交差エントロピー



# 最適化関数

optimizer = optim.SGD(net.parameters(), lr=0.01)

# optimizer = optim.Adam(net.parameters(), lr=0.0001)
for epoch in range(n_epochs):

    total_loss = 0

    

    for i, data in enumerate(train_loader):

        inputs, labels = data

        

        optimizer.zero_grad()

        

        outputs = net(inputs)

        

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()



        total_loss += loss.item()



    if (epoch+i)%60 == 0:

        print(epoch+1, total_loss)

result = torch.max(net(test_X).data, 1)[1]

accuracy = sum(test_y.data.numpy() == result.numpy()) / len(test_y.data.numpy())

accuracy