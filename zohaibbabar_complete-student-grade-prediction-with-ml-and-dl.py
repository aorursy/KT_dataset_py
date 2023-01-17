import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
data = pd.read_csv("../input/student-grade-prediction/student-mat.csv")

data.head()
data.describe()
data.info()
data.isnull().sum().any()
sc_gp = data[data['school']=='GP']['G3'].value_counts()

sc_ms = data[data['school']=='MS']['G3'].value_counts()

sc_df = pd.DataFrame([sc_gp, sc_ms], index=['School GP', 'School MS'])

sc_df = sc_df.T
sns.lineplot(data=sc_df)
fg, axs = plt.subplots(1, 2, figsize=(20,5))



axs[0].set_title('Distribution of Age & Sex wrt Score G3')

sns.barplot(x='age', y='G3', hue='sex', data=data, ax=axs[0])



axs[1].set_title('Distribution of count of Age & Sex')

sns.countplot(x='age', hue='sex', data=data, ax=axs[1])
fam_edu = data['Fedu'] + data['Medu']

sns.swarmplot(x=fam_edu, y='G3', data=data)
fg, axs = plt.subplots(1, 2, figsize=(20,5))



axs[0].set_title('Travel time wrt Score G3')

sns.swarmplot(x='traveltime', y='G3', data=data, ax=axs[0])



axs[1].set_title('Study time wrt Score G3')

sns.swarmplot(x='studytime', y='G3', data=data, ax=axs[1])
fg, axs = plt.subplots(3,3, figsize=(20,15))



axs[0,0].set_title('School Support vs G3')

sns.swarmplot(x='schoolsup', y='G3', data=data, ax=axs[0,0])

axs[0,0].set_xlabel('School Support')



axs[0,1].set_title('Family Support vs G3')

sns.swarmplot(x='famsup', y='G3', data=data, ax=axs[0,1])

axs[0,1].set_xlabel('Family Support')



axs[0,2].set_title('Paid for extra class vs G3')

sns.swarmplot(x='paid', y='G3', data=data, ax=axs[0,2])

axs[0,2].set_xlabel('Extra paid')



axs[1,0].set_title('Extra-curricular activities vs G3')

sns.swarmplot(x='activities', y='G3', data=data, ax=axs[1,0])

axs[1,0].set_xlabel('Extra-curricular activities')



axs[1,1].set_title('Nursery vs G3')

sns.swarmplot(x='nursery', y='G3', data=data, ax=axs[1,1])

axs[1,1].set_xlabel('Nursery')



axs[1,2].set_title('Higher Education vs G3')

sns.swarmplot(x='higher', y='G3', data=data, ax=axs[1,2])

axs[1,2].set_xlabel('Higher Education')



axs[2,0].set_title('Internet Access vs G3')

sns.swarmplot(x='internet', y='G3', data=data, ax=axs[2,0])

axs[2,0].set_xlabel('Internet Access')



axs[2,1].set_title('Romantic Relation vs G3')

sns.swarmplot(x='romantic', y='G3', data=data, ax=axs[2,1])

axs[2,1].set_xlabel('Romantic Relation')
#Lets check the alcohol consumption

alc = data['Dalc'] + data['Walc']

sns.swarmplot(x=alc, y='G3', data=data)
fg, axs = plt.subplots(1, 3, figsize=(20, 5))

g1 = sns.distplot(data['G1'], ax=axs[0])

g2 = sns.distplot(data['G2'], ax=axs[1])

g3 = sns.distplot(data['G3'], ax=axs[2])
fg, axs = plt.subplots(2,2, figsize=(20,10))

b1 = sns.lineplot(x='G1', y='G3', data=data, ax=axs[0,0])

b2 = sns.scatterplot(x='G1', y='G3', data=data, ax=axs[0,1])

b3 = sns.lineplot(x='G2', y='G3', data=data, ax=axs[1,0])

b4 = sns.scatterplot(x='G2', y='G3', data=data, ax=axs[1,1])
sns.countplot(x='G3', data=data, order=data['G3'].value_counts().index)
#school

sch_map = {'GP':1, 'MS':2}

data['school'] = data['school'].map(sch_map)
#sex

sex_map = {'F':1, 'M':2}

data['sex'] = data['sex'].map(sex_map)
#address

fmap = {'U':1, 'R':2}

data['address'] = data['address'].map(fmap)
#famsize

fmap = {'LE3':1, 'GT3':2}

data['famsize'] = data['famsize'].map(fmap)
#Pstatus

fmap = {'T':1, 'A':2}

data['Pstatus'] = data['Pstatus'].map(fmap)
#Mjob and Fjob

fmap = {'services':1, 'at_home':2, 'teacher':3, 'health':4, 'other':5}

data['Mjob'] = data['Mjob'].map(fmap)

data['Fjob'] = data['Fjob'].map(fmap)
#reason

fmap = {'course':1, 'home':2, 'reputation':3, 'other':4}

data['reason'] = data['reason'].map(fmap)
#guardian

fmap = {'mother':1, 'father':2, 'other':3}

data['guardian'] = data['guardian'].map(fmap)
#schoolsup famsup paid

fmap = {'yes':1, 'no':0}

data['schoolsup'] = data['schoolsup'].map(fmap)

data['famsup'] = data['famsup'].map(fmap)

data['paid'] = data['paid'].map(fmap)

data['activities'] = data['activities'].map(fmap)

data['nursery'] = data['nursery'].map(fmap)

data['higher'] = data['higher'].map(fmap)

data['internet'] = data['internet'].map(fmap)

data['romantic'] = data['romantic'].map(fmap)
X = data.iloc[:, :32]

y = data.iloc[:, -1]
from sklearn.feature_selection import SelectKBest, chi2



k_best = SelectKBest(score_func=chi2, k=15)

k_best.fit(X, y)



df_score = pd.Series(data=k_best.scores_, index=X.columns)

df_score
df_score.nlargest(15).plot(kind='bar')
plt.figure(figsize=(25, 10))

sns.heatmap(data.corr(), annot=True, cmap='YlGnBu')
data.corr()['G3'].nlargest(15)
X = data[df_score.nlargest(15).index]

y = data['G3']
from sklearn.model_selection import train_test_split, KFold, cross_val_score



k_fold = KFold(n_splits=10, random_state=10, shuffle=True)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)



X_train.shape, X_test.shape, y_train.shape, y_test.shape
from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Lasso

from sklearn.linear_model import Ridge

from sklearn.linear_model import ElasticNet

from sklearn.ensemble import RandomForestRegressor

from sklearn.svm import SVR

from sklearn.ensemble import GradientBoostingRegressor



from sklearn.metrics import mean_squared_error
classifiers = {

    'Linear Regression' : LinearRegression(),

    'Lasso': Lasso(),

    'Ridge': Ridge(),

    'ElasticNet': ElasticNet(),

    'RandromForest': RandomForestRegressor(n_estimators=100),

    'GradientBoost': GradientBoostingRegressor(n_estimators=100),

    'SVM' : SVR()

}



for key, clf in classifiers.items():

    #clf.fit(X_train, y_train)

    score = cross_val_score(clf, X_train, y_train, cv=k_fold, scoring='neg_mean_squared_error')

    rmse = np.sqrt(-score)

    rmse_score = round(np.mean(rmse), 2)

    print('RMSE score with CV of {0} is {1}'.format(key, rmse_score))
from sklearn.model_selection import GridSearchCV

clf = GradientBoostingRegressor()

params = {

    'min_samples_split':[5,9,13],'max_leaf_nodes':[3,5,7,9],'max_features':[4,5,6,7]

}

gs = GridSearchCV(estimator=clf, param_grid=params, cv=k_fold, scoring='neg_mean_squared_error')

gs.fit(X_train, y_train)
gs.best_params_
np.sqrt(-gs.best_score_)
gb_clf = gs.best_estimator_
gb_clf.fit(X_train, y_train)

y_predict = gb_clf.predict(X_test)



mse = mean_squared_error(y_test, y_predict)

rmse = np.sqrt(mse)

rmse
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

from tensorflow.keras import backend

def build_regressor():

    regressor = Sequential()

    

    regressor.add(Dense(units=15, input_dim=15, activation='relu'))

    regressor.add(Dense(units=32, activation='relu'))

    regressor.add(Dense(units=1))

    

    regressor.compile(optimizer='adam', loss='mean_squared_error',  metrics=['mae'])

    return regressor
from keras.wrappers.scikit_learn import KerasRegressor

regressor = KerasRegressor(build_fn=build_regressor, batch_size=20, epochs=100)
# Scale the train and test data before training the model

from sklearn.preprocessing import StandardScaler



ss = StandardScaler()

X_train = ss.fit_transform(X_train)

y_train = ss.fit_transform(np.array(y_train).reshape(-1, 1))



X_test = ss.fit_transform(X_test)

y_test = ss.fit_transform(np.array(y_test).reshape(-1,1))
results=regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

rmse