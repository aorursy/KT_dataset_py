import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings
df=pd.read_csv("../input/graduate-admissions/Admission_Predict_Ver1.1.csv")

df.head()
df.rename(columns={'Chance of Admit ':'chance_of_admit'},inplace=True)

df.head()
df.insert(8,"chances",0,True)

df.head()
df.loc[df['chance_of_admit']> 0.5, ['chances']] = '1'
df.dtypes
df.isnull().sum()
df.describe()
df.info()
df.shape
chances= df["chance_of_admit"].values

category = []

for num in chances:

    if num <= 0.5:

        category.append("Low")

    else:

        category.append("High")
[(i, category.count(i)) for i in set(category)]

plt.figure(figsize=(10, 6))

sns.countplot(category, palette="muted")
plt.figure(figsize=(12, 6))

sns.heatmap(df.corr(), annot=True)
df1=df.drop(['Serial No.', 'University Rating','Research','chances'], axis=1)
f,ax=plt.subplots(1,3,figsize=(25,5))

box1=sns.boxplot(data=df1["CGPA"],ax=ax[0],color='m')

ax[0].set_xlabel('CGPA')

box1=sns.boxplot(data=df1["LOR "],ax=ax[1],color='m')

ax[1].set_xlabel('LOR')

box1=sns.boxplot(data=df1["SOP"],ax=ax[2],color='m')

ax[2].set_xlabel('SOP')
f,ax=plt.subplots(1,2,figsize=(25,5))

box1=sns.boxplot(data=df1["TOEFL Score"],ax=ax[0],color='m')

ax[0].set_xlabel('TOEFL Score')

box1=sns.boxplot(data=df1["GRE Score"],ax=ax[1],color='m')

ax[1].set_xlabel('GRE Score')
import matplotlib.pyplot as plt

import seaborn as sns

df1.hist (bins=10,figsize=(20,20))

plt.show ()
sns.pairplot(df1)
sns.pairplot(df,hue = 'chances', vars = ['GRE Score','TOEFL Score','SOP','LOR ','CGPA'] )
fig=plt.figure(figsize=(10,6))

sns.countplot('University Rating',data=df )

plt.tight_layout()

plt.show()
fig=plt.figure(figsize=(8,6))

sns.countplot('Research',hue='chances',data=df )

plt.tight_layout()

plt.show()
ax = sns.distplot(df['chance_of_admit'], rug=True, hist=True)
ax = sns.violinplot(x="University Rating", y="chance_of_admit", data=df, palette="muted")
df.head(1)
df2015 = df[df.year == 2015]

trace1 = go.Box(    y=df2015["total_score"],name = 'total score of universities in 2015',

marker = dict(        color = 'rgb(200, 10, 10)',

    )

)

trace2 = go.Box(

    y=df2015["research"],

    name = 'research of universities in 2015',

    marker = dict(

        color = 'rgb(10, 200, 10)',

    )

)

    

trace3 = go.Box(

    y=df2015["citations"],

    name = 'citations of universities in 2015',

    marker = dict(color = 'rgb(10, 10, 200)',

    )

)

data = [trace1,trace2,trace3]

iplot(data)
from sklearn.model_selection import train_test_split

from sklearn import model_selection

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

import warnings

warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn import svm

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.metrics import accuracy_score

from pandas.plotting import scatter_matrix
x = df.iloc[:, 1:8] 

y=df['chances'].astype(int)

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=3)
seed=7

models = []

models.append(('RF',RandomForestClassifier()))

models.append(('SVM',SVC()))

models.append(('LR',LogisticRegression()))

models.append(('NB',GaussianNB()))

# Evaluating each models in turn

results = []

names = []

for name, model in models:

    kfold = model_selection.KFold(n_splits=10,random_state=seed)

    cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" %(name, cv_results.mean(), cv_results.std())

    print(msg)
logistic = LogisticRegression()

logistic.fit(x_train,y_train)

y_pred=logistic.predict(x_test)

print(classification_report(y_test,y_pred))

accuracy1=logistic.score(x_test,y_test)

print (accuracy1*100,'%')

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot= True)
classifier=SVC()

classifier.fit(x_train,y_train)

svm_predict=classifier.predict(x_test)

print(classification_report(y_test,svm_predict))

accuracy2=classifier.score(x_test,y_test)

print(accuracy2*100,'%')

cm = confusion_matrix(y_test, svm_predict)

sns.heatmap(cm, annot= True)
ran_class=RandomForestClassifier()

ran_class.fit(x_train,y_train)

ran_predict=ran_class.predict(x_test)

print(classification_report(y_test,ran_predict))

accuracy3=ran_class.score(x_test,y_test)

print(accuracy3*100,'%')

cm = confusion_matrix(y_test, ran_predict)

sns.heatmap(cm, annot= True)
# Defining the decision tree algorithm

from sklearn.tree import DecisionTreeClassifier

dtree=DecisionTreeClassifier()

dtree.fit(x,y)



print('Decision Tree Classifer Created')
!pip install pydotplus
# Import necessary libraries for graph viz

!pip install --upgrade scikit-learn==0.20.3

from sklearn.externals.six import StringIO  

from IPython.display import Image  

from sklearn.tree import export_graphviz

import pydotplus



# Visualize the graph

dot_data = StringIO()

export_graphviz(dtree, out_file=dot_data, feature_names=x.columns,  

                filled=True, rounded=True,

                special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  

Image(graph.create_png())