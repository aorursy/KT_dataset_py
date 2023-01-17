# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

from plotly.offline import init_notebook_mode, iplot
from plotly.graph_objs import *
import plotly.graph_objs as go
init_notebook_mode()
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
sns.set_style('darkgrid')
# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/trade-permits-current.csv')
df.columns
print("Action Information : ",df['Action Type'].unique(), "\n")
print("Worker Information : ",df['Work Type'].unique(), "\n ")
print("Contractor Information : ",df['Contractor'].unique(), len(df['Contractor'].unique()), "\n")
print("Categorical Information :", df['Category'].unique())
df.dropna(inplace = True)
df.head()
mySummingGroup = df.drop(columns=['Longitude', 'Latitude', 'Application/Permit Number']).groupby(by = 'Contractor').agg({'Value':sum})
x = mySummingGroup['Value'].nlargest(10)
x

data1 = [Bar(
            y=x,
            x=x.keys(),
            marker = dict(
            color = 'rgba(25, 82, 1, .9)'
            ),
            name = "Contractor's amount earned per project"
    )]

layout1 = go.Layout(
    title="Top Grossing Contractors",
    xaxis=dict(
        title='Contractor',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    ),
    yaxis=dict(
        title='Total Amount Earned',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    )
)
myFigure2 = go.Figure(data = data1 , layout = layout1)
iplot(myFigure2)
myMeanGroup = df.drop(columns=['Longitude', 'Latitude', 'Application/Permit Number']).groupby(by = 'Contractor').mean()
efficientContractors = myMeanGroup['Value'].nlargest(10)
data = [Bar(
            y=efficientContractors,
            x=efficientContractors.keys(),
            marker = dict(
            color = 'rgba(255, 182, 1, .9)'
            ),
            name = "Contractor's amount earned per project"
    )]

layout = go.Layout(
    title="Contractor's amount earned per project",
    xaxis=dict(
        title='Contractor',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    ),
    yaxis=dict(
        title='Amount per project',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    )
)
myFigure = go.Figure(data = data , layout = layout)
iplot(myFigure)
catCount = df.groupby('Category')['Permit Type'].count()
fig = { 
    "data":[{
        "values":catCount,
        "labels":catCount.keys(),
        "domain": {"x": [0, 1]},
        "name": "Categories",
        "hoverinfo":"label+percent+name",
        "hole": .4,
        "type": "pie",
        "textinfo": "value"
    }],
    "layout":{
        "title":"Categorical Distribution of Tenders",
        "annotations": [
            {
                "font": {
                    "size": 15
                },
                "showarrow": False,
                "text": "DISTRIBUTION",
                "x": 0.5,
                "y": 0.5
            }]
    }
}

trace = go.Pie(labels = catCount.keys(), values=catCount,textinfo='value', hoverinfo='label+percent', textfont=dict(size = 15))
iplot(fig)

# My Value Encoder
def valueEncoder(value):
    if value > 10000000:
        return 4
    elif value > 100000:
        return 3
    elif value > 10000:
        return 2
    elif value > 100:
        return 1
    else:
        return 0
df['ValueLabel'] = df['Value'].apply(valueEncoder)
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

genLabel_cat = LabelEncoder()
cat_labels = genLabel_cat.fit_transform(df['Category'])
df['CategoryLabel'] = cat_labels

df[['Category','CategoryLabel']].iloc[::2]
cat_ohe = OneHotEncoder()
cat_feature_arr = cat_ohe.fit_transform(df[['CategoryLabel']]).toarray()
cat_feature_labels = list(genLabel_cat.classes_)
cat_features = pd.DataFrame(cat_feature_arr, columns=cat_feature_labels)
cat_features.head(10)
final_one_hot = pd.get_dummies(df['Category'])
df2 = pd.concat([df, final_one_hot], axis = 1)
df2 = df2.drop(['Application/Permit Number', 'Address', 'Description', 'Applicant Name','Application Date','Issue Date','Final Date','Expiration Date','Contractor', 'Permit and Complaint Status URL', 'Location'], axis = 1)
# also add 'Value', 'Category' when running for first time
df2 = df2.drop(['CategoryLabel'],axis = 1)
df2.head()
df2 = pd.concat([df2, pd.get_dummies(df['Work Type'])], axis = 1)
df2 = df2.drop(['Work Type'], axis = 1)

df2.head()
print(df2['Action Type'].unique() , "\n Total types are ", len(df2['Action Type'].unique()))
from sklearn.feature_extraction import FeatureHasher

fh = FeatureHasher(n_features = 5, input_type = 'string')
hashed_features = fh.fit_transform(df2['Action Type'])
hashed_features = hashed_features.toarray()
df2 = pd.concat([df2, pd.DataFrame(hashed_features)], 
          axis=1).dropna()
df2.iloc[10:20]
df2['Status'].unique()
# Again, a binary parameter, let's use binary encodings.

df2 = pd.concat([df2, pd.get_dummies(df2['Status'])], axis = 1)
df2 = df2.drop(['Status'], axis = 1)
df2.drop(['Value', 'Category'], axis = 1, inplace = True)
df2.drop(['Action Type'], axis = 1, inplace = True)
df2.drop(['Permit Type'], axis = 1, inplace = True)
df2.head()

from sklearn.model_selection import train_test_split
y = df2['ValueLabel']
X = df2.drop(['ValueLabel'], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
from sklearn.naive_bayes import GaussianNB
myClassifier = GaussianNB()
myClassifier.fit(X_train, y_train)
predictions = myClassifier.predict(X_test)
from sklearn.metrics import accuracy_score,confusion_matrix

cnf = confusion_matrix(y_test, predictions)
score = accuracy_score(y_test, predictions)

print ("Confusion Matrix for our Naive Bayes classifier is :\n ", cnf)

print("While the accuracy score for the same is %.2f percent" % (score * 100))
from sklearn.tree import DecisionTreeClassifier
myClassifier2 = DecisionTreeClassifier(max_depth = 5, min_samples_leaf = 2)

myClassifier2.fit(X_train, y_train)
predictions2 = myClassifier2.predict(X_test)

cnf2 = confusion_matrix(y_test, predictions2)
score2 = accuracy_score(y_test, predictions2)

print ("Confusion Matrix for our Decision Tree classifier is :\n ", cnf2)

print("While the accuracy score for the same is %.2f percent" % (score2 * 100))
