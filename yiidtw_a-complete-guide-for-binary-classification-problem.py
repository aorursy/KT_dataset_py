import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
%matplotlib inline
train_csv = pd.read_csv('../input/train.csv')
final_csv = pd.read_csv('../input/test.csv')
train_csv.columns
def show_null_count(csv):
    idx = csv.isnull().sum()
    idx = idx[idx>0]
    idx.sort_values(inplace=True)
    idx.plot.bar()
def get_corr(col, csv):
    corr = csv.corr()[col]
    idx_gt0 = corr[corr>0].sort_values(ascending=False).index.tolist()
    return corr[idx_gt0]

show_null_count(train_csv)
show_null_count(final_csv)
sns.heatmap(train_csv.corr(), vmax=.8, square=True)
print(get_corr('Survived', train_csv))
import re
# Define function to extract titles from passenger names
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""
import re
def get_simplified_title(csv):
    def get_title(name):
        title_search = re.search(' ([A-Za-z]+)\.', name)
        if title_search:
            return title_search.group(1)
        return ""
    title = csv['Name'].apply(get_title)
    sim_title = title.replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    sim_title = sim_title.replace('Mlle', 'Miss')
    sim_title = sim_title.replace('Ms', 'Miss')
    sim_title = sim_title.replace('Mme', 'Mrs')
    return sim_title
train_csv['FamilyCount'] = train_csv['SibSp'] + train_csv['Parch'] + 1
train_csv['SimplifiedTitle'] = get_simplified_title(train_csv)
train_csv['SimplifiedTitle'].unique()
train_csv['Age'].fillna(train_csv['Age'].median(), inplace=True)
train_csv['AgeBin'] = pd.cut(train_csv['Age'], bins=[0,12,20,40,120], labels=['Children','Teenage','Adult','Elder'])
train_csv['Embarked'].fillna(train_csv['Embarked'].mode()[0], inplace = True)
train_csv['Fare'].fillna(train_csv['Fare'].median(), inplace = True)
train_csv['FareBin'] = pd.cut(train_csv['Fare'], bins=[-1, 
                                                  train_csv['Fare'].quantile(.25),
                                                  train_csv['Fare'].quantile(.5), 
                                                  train_csv['Fare'].quantile(.75),
                                                  train_csv['Fare'].max()],
                                                labels=['LowFare', 
                                                        'MediumFare',
                                                        'HighFare', 
                                                        'TopFare'])
train_csv.columns
train_df = train_csv.copy()
train_df.drop(['PassengerId', 'Name', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'FamilyCount'], axis=1, inplace=True)
train_df.head(n=5)
train_df = pd.get_dummies(train_df, columns = ["Pclass", "Sex","Embarked","SimplifiedTitle","AgeBin","FareBin"],
                            prefix=["PC", "Sex","Em","ST","Age","Fare"])
train_df.head(n=5)
sns.heatmap(train_df.corr(),annot=True,cmap='RdYlGn',linewidths=0.2)
fig=plt.gcf()
fig.set_size_inches(20,12)
plt.show()
def throttling(arr, thres):
    #res = arr.copy()
    res = np.zeros(len(arr))
    res[arr >= thres] = int(1)
    res[arr < thres] = int(0)
    return res
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(train_df.drop('Survived', axis=1),
                                                 train_df['Survived'],
                                                 test_size=0.2,
                                                 random_state=123)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
lr = LogisticRegression()
lr.fit(x_train,y_train)
y_pred_lr = lr.predict(x_test)
print('The accuracy of the Logistic Regression is',round(accuracy_score(y_pred_lr,y_test)*100,2))
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
def baselineNN(dims):
    model = Sequential()
    model.add(Dense(10, input_dim=dims, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    #model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
def use_keras_nn_model(x, y, xx, yy, epochs):
    model = baselineNN(x.shape[1])
    model.fit(x.as_matrix(), y.as_matrix(), epochs=epochs)
    y_pred = model.predict(xx.as_matrix()).reshape(xx.shape[0],)
    return y_pred, model
y_pred_nn, model_nn = use_keras_nn_model(x_train, y_train, x_test, y_test, 100)
#print('The accuracy of the Neural Network is',round(accuracy_score(y_pred_nn_thres,y_test)*100,2))
print('The accuracy of the Neural Network is',round(accuracy_score(throttling(y_pred_nn, 0.6), y_test)*100,2))
import xgboost as xgb
from xgboost import plot_importance
params = {
    'objective': 'binary:logistic',
    'gamma': 0.1,
    'max_depth': 5,
    'lambda': 3,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 3,
    'silent': 1,
    'eta': 0.1,
    'seed': 1000,
    'nthread': 4,
}

num_round = 10
dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(x_test, label=y_test)
watchlist = [(dtrain, 'train'), (dtest, 'test')]
bst = xgb.train(params, dtrain, num_round, watchlist)
y_pred_xgb = bst.predict(dtest)
print('The accuracy of the Neural Network is',round(accuracy_score(throttling(y_pred_xgb, 0.6),y_test)*100,2))
plot_importance(bst)

final_csv.columns
final_csv['Fare'].fillna(final_csv['Fare'].median(), inplace = True)
final_csv['Age'].fillna(final_csv['Age'].median(), inplace=True)
final_csv['AgeBin'] = pd.cut(final_csv['Age'], bins=[0,12,20,40,120], labels=['Children','Teenage','Adult','Elder'])
final_csv['FamilyCount'] = final_csv['SibSp'] + final_csv['Parch'] + 1
final_csv['SimplifiedTitle'] = get_simplified_title(final_csv)
final_csv['FareBin'] = pd.cut(final_csv['Fare'], bins=[-1, 
                                                  final_csv['Fare'].quantile(.25),
                                                  final_csv['Fare'].quantile(.5), 
                                                  final_csv['Fare'].quantile(.75),
                                                  final_csv['Fare'].max()],
                                                labels=['LowFare', 
                                                        'MediumFare',
                                                        'HighFare', 
                                                        'TopFare'])
final_df = final_csv.copy()
final_df.drop(['PassengerId', 'Name', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'FamilyCount'], axis=1, inplace=True)
final_df.head(n=5)
final_df = pd.get_dummies(final_df, columns = ["Pclass", "Sex","Embarked","SimplifiedTitle","AgeBin","FareBin"],
                            prefix=["PC", "Sex","Em","ST","Age","Fare"])
final_df.head(n=5)
y_final_prob = model_nn.predict(final_df.as_matrix()).reshape(final_df.shape[0],)
y_final = throttling(y_final_prob, .6)
summission = pd.concat([final_csv['PassengerId'], pd.DataFrame(y_final)], axis=1)
summission.columns = ['PassengerId', 'Survived']
summission.to_csv('summission.csv', encoding='utf-8', index = False)
