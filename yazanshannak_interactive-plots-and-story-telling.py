#Import common modules

import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import seaborn as sns

import matplotlib

import warnings

warnings.filterwarnings("ignore", message="numpy.dtype size changed")

warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly.tools as tls

import plotly.graph_objs as go

# init_notebook_mode(connected=True)

import plotly

# plotly.tools.set_credentials_file(username='YazanSh', api_key='2GxEQaYp9s6UJAc0bpY5')

# cf.go_offline()





%matplotlib inline
data = pd.read_csv('../input/train.csv')

data.head()
data.columns.to_list()
train = data.copy()
train.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)
missing = pd.DataFrame(train.count(), columns=['Count'])

missing['Missing Values'] = 891 - missing['Count']

missing['%Missing Values'] = (missing['Missing Values'] / 891) * 100

missing.drop('Count', axis=1, inplace=True)

missing = missing[missing['Missing Values'] != 0]

missing
sns.set(rc={'figure.figsize':(11.7, 8.27)})

sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')

plt.show()
train.drop(['Cabin'], axis=1, inplace=True)
train.dropna(axis=0, subset=['Embarked'], inplace=True)
# Preparing Data for plots

#General

train.replace({'S':'Southampton', 'C':'Cherbourg', 'Q':'Queenstown'}, inplace=True)

embarked_points = ['Cherbourg', 'Queenstown', 'Southampton']

#Total

embarked_count_total = train.groupby(['Embarked'], as_index=True, sort=True).count()['Survived'].to_list()

#Sruvived

embarked_count_survived = train[train['Survived'] == 1].groupby(['Embarked'], as_index=True, sort=True).count()['Survived']

embarked_count_died = train[train['Survived'] == 0].groupby(['Embarked'], as_index=True, sort=True).count()['Survived']

#Sex

embarked_count_male = train[train['Sex'] == 'male'].groupby(['Embarked'], as_index=True, sort=True).count()['Sex']

embarked_count_female = train[train['Sex'] == 'female'].groupby(['Embarked'], as_index=True, sort=True).count()['Sex']

#Passenger Class

embarked_count_Pclass1 = train[train['Pclass'] == 1].groupby(['Embarked'], as_index=True, sort=True).count()['Pclass']

embarked_count_Pclass2 = train[train['Pclass'] == 2].groupby(['Embarked'], as_index=True, sort=True).count()['Pclass']

embarked_count_Pclass3 = train[train['Pclass'] == 3].groupby(['Embarked'], as_index=True, sort=True).count()['Pclass']





# Preparing figure objects

#Total Pie Chart

embarked_pie = go.Pie(labels=embarked_points, values=embarked_count_total)

embarked_pie_layout = go.Layout(title='Embarked')

embarked_pie_fig = go.Figure(data = [embarked_pie], layout=embarked_pie_layout)

#Survived Stacked Bar Chart

embarked_bar_died = go.Bar(x=embarked_points, y=embarked_count_died, name="Didn't Survive")

embarked_bar_survived = go.Bar(x=embarked_points, y=embarked_count_survived, name="Survived")

embarked_bar_layout = go.Layout(barmode='stack', title='Embark Points and Survival')

embarked_bar_fig = go.Figure(data=[embarked_bar_died, embarked_bar_survived], layout=embarked_bar_layout)

#Age Box plots

Q_age = go.Box(y=train[train['Embarked'] == 'Queenstown']['Age'], name='Queenstown')

C_age = go.Box(y=train[train['Embarked'] == 'Cherbourg']['Age'], name='Cherbourg')

S_age = go.Box(y=train[train['Embarked'] == 'Southampton']['Age'], name='Southampton')

age_box_layout = go.Layout(title='Age amongst Embark Points')

age_box_fig = go.Figure(data=[C_age, Q_age, S_age ], layout=age_box_layout)

#Sex Stacked Bar Chart

embarked_bar_female = go.Bar(x=embarked_points, y=embarked_count_female, name='Female')

embarked_bar_male = go.Bar(x=embarked_points, y=embarked_count_male, name='Male')

embarked_sex_layout = go.Layout(title='Embarked and Sex', barmode='stack')

embarked_sex_fig = go.Figure(data=[embarked_bar_female, embarked_bar_male], layout=embarked_sex_layout)

#Fare Box Plot

Q_fare = go.Box(y=train[train['Embarked'] == 'Queenstown']['Fare'], name='Queenstown')

C_fare = go.Box(y=train[train['Embarked'] == 'Cherbourg']['Fare'], name='Cherbourg')

S_fare = go.Box(y=train[train['Embarked'] == 'Southampton']['Fare'], name='Southampton')

fare_box_layout = go.Layout(title='Embarkment and Fares')

fare_box_fig = go.Figure(data=[C_fare, Q_fare, S_fare ], layout=fare_box_layout)



#PClass Grouped Bar Plot

embarked_bar_Pclass1 = go.Bar(x=embarked_points, y=embarked_count_Pclass1, name='Upper Class')

embarked_bar_Pclass2 = go.Bar(x=embarked_points, y=embarked_count_Pclass2, name='Middle Class')

embarked_bar_Pclass3 = go.Bar(x=embarked_points, y=embarked_count_Pclass3, name='Lower Class')

embarked_Pclass_layout = go.Layout(title='Embarkment and Passenger Class')

embarked_Pclass_fig = go.Figure(data = [embarked_bar_Pclass1, embarked_bar_Pclass2, embarked_bar_Pclass3], layout=embarked_Pclass_layout)





# Plotting

iplot(embarked_pie_fig, filename='Titanic Embarked', )

iplot(embarked_bar_fig)

iplot(age_box_fig)

iplot(embarked_sex_fig)

iplot(fare_box_fig)

iplot(embarked_Pclass_fig)
#Preparing the data

#General

all_fares = train['Fare']

#Survival

fares_survived = train[train['Survived'] == 1]['Fare']

fares_died = train[train['Survived'] == 0]['Fare']





#Preparing figures

#General

all_fares_hist = go.Histogram(x=all_fares, marker={'color':'#35477d'})

all_fares_hist_layout = go.Layout(title='Fares')

all_fares_hist_fig = go.Figure(data=[all_fares_hist], layout=all_fares_hist_layout)

#Survival

fares_survived_box = go.Box(y=fares_survived, name='Survived', marker={'color':'#0939CF'})

fares_died_box = go.Box(y=fares_died, name="Didn't Survive", marker={'color':'#CF1E09'})

fares_survival_layout = go.Layout(title='Fares and Survival')

fares_survival_fig = go.Figure(data=[fares_died_box, fares_survived_box], layout=fares_survival_layout)





#Plotting

iplot(all_fares_hist_fig)

iplot(fares_survival_fig)
from sklearn.preprocessing import Imputer

imputer = Imputer()

train['Age'] = imputer.fit_transform(train['Age'].to_numpy().reshape(-1,1))
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

train['Fare'] = scaler.fit_transform(train['Fare'].to_numpy().reshape(-1,1))

train['Age'] = scaler.fit_transform(train['Age'].to_numpy().reshape(-1,1))
encoded_sex =  pd.get_dummies(train['Sex'])

encoded_class = pd.get_dummies(train['Pclass'], prefix='Pclass')

encoded_embark = pd.get_dummies(train['Embarked'], prefix='embarked_from')

train = pd.concat([train, encoded_sex, encoded_class, encoded_embark], axis=1)



try:

    train.drop('Sex', axis=1, inplace=True)

except:

    pass

try:

    train.drop('Pclass', axis=1, inplace=True)

except:

    pass

try:

    train.drop('Embarked', axis=1, inplace=True)

except:

    pass
y = train['Survived']

X = train.drop('Survived', axis=1)

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y)
from sklearn.neighbors import KNeighborsClassifier

KNN = KNeighborsClassifier()

## Grid search to find optimal number of neighbours

from sklearn.model_selection import GridSearchCV

search = GridSearchCV(estimator=KNN, param_grid={'n_neighbors':range(1,15)}, scoring='precision', n_jobs=-1, cv=10)

search.fit(X_train, y_train)

KNN_optimum = search.best_estimator_
## KNN with grid search result

predictions = KNN_optimum.predict(X_test)

from sklearn.metrics import classification_report

KNN_report = classification_report(y_test, predictions)

print(KNN_report)
from sklearn.svm import SVC

SVM = SVC()

SVM.fit(X_train, y_train)

SVM_predictions = SVM.predict(X_test)

print(classification_report(y_test, SVM_predictions))
from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier()

RF.fit(X_train, y_train)

RF_predictions = RF.predict(X_test)

print(classification_report(y_test, RF_predictions))
from sklearn.ensemble import VotingClassifier

VC = VotingClassifier(estimators=[('rf', RF), ('svm', SVM), ('knn', KNN_optimum)])
VC.fit(X_train, y_train)

VC_predictions = VC.predict(X_test)

print(classification_report(y_test, VC_predictions))