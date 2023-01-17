import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly
import seaborn as sns
import plotly.express as px
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error,accuracy_score,confusion_matrix,recall_score,precision_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import plotly.graph_objects as go
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
dataset = pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict.csv', index_col= ['Serial No.'])
dataset.head(5)
dataset.rename(columns={'GRE Score':"GRE", "TOEFL Score":"TOEFL","University Rating":"Uni_Rating",'Chance of Admit ':'COA'}, inplace = True)
dataset.info()
fig = px.histogram(dataset, x="GRE", y="GRE", marginal="rug",
                   hover_data=dataset.columns ,width=800, height=400)
fig.show()
fig = px.histogram(dataset, x="TOEFL", y="TOEFL", marginal="rug",
                   hover_data=dataset.columns ,width=800, height=400)
fig.show()
fig = px.histogram(dataset, x="COA", y="COA", marginal="rug",
                   hover_data=dataset.columns ,width=800, height=400)
fig.show()
fig = px.histogram(dataset, x="CGPA", y="CGPA", marginal="rug",
                   hover_data=dataset.columns ,width=800, height=400)
fig.show()
heatmap1 = dataset.corr()
sns.heatmap(heatmap1,annot = True,vmin =.3, vmax = 1)
high_chances = dataset[dataset['COA']>0.8]
low_chances = dataset[dataset['COA']<0.6]
medium_chances = dataset[dataset['COA'].between(0.6,0.8)]
fig = go.Figure()
fig.add_trace(go.Box(
    y=high_chances['GRE'].values,
    x=dataset['Uni_Rating'].values,
    name='applicants with higher chances of admission',
    marker_color='#32a83c'
))

fig.add_trace(go.Box(
    y=low_chances['GRE'].values,
    x=dataset['Uni_Rating'].values,
    name='applicants with lower chances of admission',
    marker_color='#ff4242'
))

fig.add_trace(go.Box(
    y=medium_chances['GRE'].values,
    x=dataset['Uni_Rating'].values,
    name='applicants with mediocre chances of admission',
    marker_color='#fca503'
))

fig.update_layout(
    yaxis_title='GRE Scores',
    boxmode='group', # group together boxes of the different traces for each value of x
    title = 'GRE Score distribution across university ratings',
    xaxis_title = 'University ratings'
)
fig.show()
fig = go.Figure()
fig.add_trace(go.Box(
    y=high_chances['TOEFL'].values,
    x=dataset['Uni_Rating'].values,
    name='applicants with higher chances of admission',
    marker_color='#32a83c'
))

fig.add_trace(go.Box(
    y=low_chances['TOEFL'].values,
    x=dataset['Uni_Rating'].values,
    name='applicants with lower chances of admission',
    marker_color='#ff4242'
))

fig.add_trace(go.Box(
    y=medium_chances['TOEFL'].values,
    x=dataset['Uni_Rating'].values,
    name='applicants with mediocre chances of admission',
    marker_color='#fca503'
))

fig.update_layout(
    yaxis_title='TOEFL score',
    boxmode='group' ,# group together boxes of the different traces for each value of x
    title = 'TOEFL Score distribution across university ratings'
    ,xaxis_title = 'University ratings'
)
fig.show()
fig = go.Figure()
fig.add_trace(go.Box(
    y=high_chances['CGPA'].values,
    x=dataset['Uni_Rating'].values,
    name='applicants with higher chances of admission',
    marker_color='#32a83c'
))

fig.add_trace(go.Box(
    y=low_chances['CGPA'].values,
    x=dataset['Uni_Rating'].values,
    name='applicants with lower chances of admission',
    marker_color='#ff4242'
))

fig.add_trace(go.Box(
    y=medium_chances['CGPA'].values,
    x=dataset['Uni_Rating'].values,
    name='applicants with mediocre chances of admission',
    marker_color='#fca503'
))

fig.update_layout(
    yaxis_title='CGPA',
    boxmode='group',
    title = 'CGPA distribution across university ratings'
    ,xaxis_title = 'University ratings'
)
fig.show()
fig = go.Figure()
fig.add_trace(go.Box(
    y=high_chances['SOP'].values,
    x=dataset['Uni_Rating'].values,
    name='applicants with higher chances of admission',
    marker_color='#32a83c'
))

fig.add_trace(go.Box(
    y=low_chances['SOP'].values,
    x=dataset['Uni_Rating'].values,
    name='applicants with lower chances of admission',
    marker_color='#ff4242'
))

fig.add_trace(go.Box(
    y=medium_chances['SOP'].values,
    x=dataset['Uni_Rating'].values,
    name='applicants with mediocre chances of admission',
    marker_color='#fca503'
))

fig.update_layout(
    yaxis_title='SOP',
    boxmode='group',
    title = 'SOP distribution across university ratings'
    ,xaxis_title = 'University ratings'
)
fig.show()
X = dataset.loc[:,'GRE':'Research']
Y = dataset.loc[:,'COA']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3,shuffle = True)
LR = LinearRegression()
LR.fit(X_train,y_train)
ypred = LR.predict(X_test)
fig = px.scatter(x=y_test, y=ypred, trendline = 'ols')
fig.show()
print ("Mean absolute error is:", mean_absolute_error(y_test,ypred).round(3))
print ("R2 Score is:", r2_score(y_test,ypred).round(3))
print ("Mean Squared error is:", mean_squared_error(y_test,ypred).round(3))
low = dataset['COA'] <0.6
mediocre = dataset['COA'].between(0.6,0.8)
high = dataset['COA'].between(0.8,1)
dataset.loc[low,'COA'] = "LOW"
dataset.loc[mediocre,'COA'] = "MEDIOCRE"
dataset.loc[high,'COA'] = "HIGH"
LE = LabelEncoder()
dataset['COA'] = LE.fit_transform(dataset['COA'].values)
LR = LogisticRegression()
Xclf = dataset.loc[:,'GRE':'Research']
Yclf = dataset.loc[:,'COA']
X_train, X_test, y_train, y_test = train_test_split(Xclf, Yclf, test_size=0.3,shuffle = True)
LR.fit(X_train,y_train)
ypred = LR.predict(X_test)
print("accuracy score is: ", accuracy_score(y_test,ypred))
cnf = confusion_matrix(LE.inverse_transform(y_test),LE.inverse_transform(ypred))
cnf_df = pd.DataFrame(cnf, index =['Low','Mediocre','High'], columns = ['Low','Mediocre','High'])
sns.heatmap(cnf_df, annot = True)