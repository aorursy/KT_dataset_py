import numpy as np

import pandas as pd



import plotly.graph_objects as go

import plotly.express as px

import plotly.io as pio

pio.templates.default = "plotly_dark"

from plotly.subplots import make_subplots

import folium 

from folium import plugins

from tqdm.notebook import tqdm as tqdm





from pathlib import Path

data_dir = Path('../input/pima-indians-diabetes-database')





import warnings

warnings.filterwarnings('ignore')



import matplotlib.pyplot as plt

import seaborn as sns



from scipy.stats import skew   

import pylab as p  

  

from pandas.plotting import scatter_matrix



import xgboost as xgb

from sklearn.metrics import mean_squared_error





from mlxtend.plotting import plot_decision_regions

from sklearn import metrics

data = pd.read_csv(data_dir/'diabetes.csv')

data.head()
data.info()
data.describe()
data1=data.copy(deep=True)

data1[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = data1[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)

print(data1.isnull().sum())
fig = make_subplots(rows=3, cols=3, subplot_titles=('Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',

       'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'))



trace0= go.Histogram(

    

    x=data['Pregnancies'],

    name="Pregnancies",

    opacity=0.75

)



trace1= go.Histogram(

    

    x=data['Glucose'],

    name="Glucose",

    opacity=0.75

)



trace2= go.Histogram(

    

    x=data['BloodPressure'],

    name="BloodPressure",

    opacity=0.75

)



trace3= go.Histogram(

    

    x=data['SkinThickness'],

    name="SkinThickness",

    opacity=0.75

)



trace4= go.Histogram(

    

    x=data['Insulin'],

    name="Insulin",

    opacity=0.75

)



trace5= go.Histogram(

    

    x=data['BMI'],

    name="BMI",

    opacity=0.75

)



trace6= go.Histogram(

    

    x=data['DiabetesPedigreeFunction'],

    name="DiabetesPedigreeFunction",

    opacity=0.75

)



trace7= go.Histogram(

    

    x=data['Age'],

    name="Age",

    opacity=0.75

)



trace8= go.Histogram(

    

    x=data['Outcome'],

    name="Outcome",

    opacity=0.75

)



fig.append_trace(trace0,1,1)

fig.append_trace(trace1,1,2)

fig.append_trace(trace2,1,3)

fig.append_trace(trace3,2,1)

fig.append_trace(trace4,2,2)

fig.append_trace(trace5,2,3)

fig.append_trace(trace6,3,1)

fig.append_trace(trace7,3,2)

fig.append_trace(trace8,3,3)



fig.update_layout(template="plotly_dark",title_text='<b>Visualization before dealing Nan values</b>',font=dict(family="Arial,Balto,Courier new,Droid sans",color='white'))

fig.show()
data1['Glucose'].fillna(data1['Glucose'].mean(), inplace = True)

data1['BloodPressure'].fillna(data1['BloodPressure'].mean(), inplace = True)

data1['SkinThickness'].fillna(data1['SkinThickness'].median(), inplace = True)

data1['Insulin'].fillna(data1['Insulin'].median(), inplace = True)

data1['BMI'].fillna(data1['BMI'].median(), inplace = True)
fig = make_subplots(rows=3, cols=3, subplot_titles=('Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',

       'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'))



trace0= go.Histogram(

    

    x=data1['Pregnancies'],

    name="Pregnancies",

    opacity=0.75

)



trace1= go.Histogram(

    

    x=data1['Glucose'],

    name="Glucose",

    opacity=0.75

)



trace2= go.Histogram(

    

    x=data1['BloodPressure'],

    name="BloodPressure",

    opacity=0.75

)



trace3= go.Histogram(

    

    x=data1['SkinThickness'],

    name="SkinThickness",

    opacity=0.75

)



trace4= go.Histogram(

    

    x=data1['Insulin'],

    name="Insulin",

    opacity=0.75

)



trace5= go.Histogram(

    

    x=data1['BMI'],

    name="BMI",

    opacity=0.75

)



trace6= go.Histogram(

    

    x=data1['DiabetesPedigreeFunction'],

    name="DiabetesPedigreeFunction",

    opacity=0.75

)



trace7= go.Histogram(

    

    x=data1['Age'],

    name="Age",

    opacity=0.75

)



trace8= go.Histogram(

    

    x=data1['Outcome'],

    name="Outcome",

    opacity=0.75

)



fig.append_trace(trace0,1,1)

fig.append_trace(trace1,1,2)

fig.append_trace(trace2,1,3)

fig.append_trace(trace3,2,1)

fig.append_trace(trace4,2,2)

fig.append_trace(trace5,2,3)

fig.append_trace(trace6,3,1)

fig.append_trace(trace7,3,2)

fig.append_trace(trace8,3,3)



fig.update_layout(template="plotly_dark",title_text='<b>Visualization After NaN removal</b>',font=dict(family="Arial,Balto,Courier new,Droid sans",color='white'))

fig.show()
data1.skew(axis=0)
plot=scatter_matrix(data,figsize=(20, 20))
plot=sns.pairplot(data, hue = 'Outcome')
plt.figure(figsize=(10,8))  

heatmap=sns.heatmap(data.corr(), annot=True,cmap ='RdYlGn') 

heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12);
plt.figure(figsize=(10,8))  

heatmap=sns.heatmap(data1.corr(), annot=True,cmap ='RdYlGn') 

heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12);
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X =  pd.DataFrame(sc_X.fit_transform(data1.drop(["Outcome"],axis = 1),),

columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI', 'DiabetesPedigreeFunction', 'Age'])
X.head()
y = data1.Outcome
data_dmatrix = xgb.DMatrix(data=X,label=y)
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42, stratify=y)
xg_reg = xgb.XGBRegressor(objective ='binary:logistic', colsample_bytree = 0.3, learning_rate = 0.1,

                max_depth = 5, alpha = 10, n_estimators = 10)
xg_reg.fit(X_train,y_train)
preds = xg_reg.predict(X_test)
preds
preds1=[]

n= len(preds)

for i in range(n):

    if(preds[i]>=0.5):

        preds1.append(1)

    else:

        preds1.append(0)
preds1
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, preds1)
from sklearn.neighbors import KNeighborsClassifier





test_scores = []

train_scores = []



for i in range(1,15):



    knn = KNeighborsClassifier(i)

    knn.fit(X_train,y_train)

    

    train_scores.append(knn.score(X_train,y_train))

    test_scores.append(knn.score(X_test,y_test))
max_train_score = max(train_scores)

train_scores_ind = [i for i, v in enumerate(train_scores) if v == max_train_score]

print('Max train score {} % and k = {}'.format(max_train_score*100,list(map(lambda x: x+1, train_scores_ind))))
max_test_score = max(test_scores)

test_scores_ind = [i for i, v in enumerate(test_scores) if v == max_test_score]

print('Max test score {} % and k = {}'.format(max_test_score*100,list(map(lambda x: x+1, test_scores_ind))))
plt.figure(figsize=(12,5))

p = sns.lineplot(range(1,15),train_scores,marker='*',label='Train Score')

p = sns.lineplot(range(1,15),test_scores,marker='o',label='Test Score')
knn = KNeighborsClassifier(11)



knn.fit(X_train,y_train)

knn.score(X_test,y_test)
value = 20000

width = 20000

plot_decision_regions(X.values, y.values, clf=knn, legend=2, 

                      filler_feature_values={2: value, 3: value, 4: value, 5: value, 6: value, 7: value},

                      filler_feature_ranges={2: width, 3: width, 4: width, 5: width, 6: width, 7: width},

                      X_highlight=X_test.values)





plt.title('KNN with Diabetes Data')

plt.show()
y_pred = knn.predict(X_test)

confusion_matrix(y_test,y_pred)

pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

p = sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')

plt.title('Confusion matrix', y=1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')
from sklearn.metrics import classification_report

print(classification_report(y_test,y_pred))
from sklearn.metrics import roc_curve

y_pred_proba = knn.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)




plt.plot([0,1],[0,1],'k--')

plt.plot(fpr,tpr, label='Knn')

plt.xlabel('fpr')

plt.ylabel('tpr')

plt.title('Knn(n_neighbors=11) ROC curve')

plt.show()



from sklearn.metrics import roc_auc_score

roc_auc_score(y_test,y_pred_proba)
from sklearn.model_selection import GridSearchCV

#In case of classifier like knn the parameter to be tuned is n_neighbors

param_grid = {'n_neighbors':np.arange(1,50)}

knn = KNeighborsClassifier()

knn_cv= GridSearchCV(knn,param_grid,cv=5)

knn_cv.fit(X,y)



print("Best Score:" + str(knn_cv.best_score_))

print("Best Parameters: " + str(knn_cv.best_params_))