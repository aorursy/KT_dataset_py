import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#!pip install chart_studio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#import plotly.graph_objs as go
#import chart_studio.plotly as py
#import plotly.offline as pyo
#import chart_studio.plotly as py

df = pd.read_csv('../input/av-janatahack-machine-learning-in-agriculture/train_yaOffsB.csv')
df_test = pd.read_csv('../input/av-janatahack-machine-learning-in-agriculture/test_pFkWwen.csv')
df_original = df.copy()
df_test_original = df_test.copy()
print(df.shape)
df.head()
print(df_test.shape)
df_test.head()
df.describe()
df.dtypes
cat_col = ['Crop_Type','Soil_Type','Pesticide_Use_Category','Season']
num_col = ['Estimated_Insects_Count','Number_Doses_Week','Number_Weeks_Used','Number_Weeks_Quit']
tag_col = 'Crop_Damage'

df[tag_col].value_counts()
# data = [go.Histogram(x = df[tag_col])]
# layout = go.Layout(title = 'Crop Damage')
# fig = go.Figure(data = data, layout = layout)
# py.iplot(fig, filename='jupyter-basic_bar')
#pyo.plot(fig)
df[tag_col].value_counts().plot.bar()
plt.figure(1)
subplotplace = 221
for col in cat_col:
    plt.subplot(subplotplace)
    df[col].value_counts(normalize = True).plot.bar(figsize = (20,10),title = col.replace('_',' '))
    subplotplace += 1
plt.figure(1,figsize = (10,20))
subplotplace = 421
for col in num_col:
    plt.subplot(subplotplace)
    sns.distplot(df[col])
    subplotplace += 1
    plt.subplot(subplotplace)
    df[col].plot.box()
    subplotplace += 1
    
#plt.scatter(df['ID'], df['Estimated_Insects_Count'])
df.boxplot(column = ['Estimated_Insects_Count'], by = 'Soil_Type')
plt.figure(1)
#subplotplace = 221
for col in cat_col:
    p =pd.crosstab(df[col],df[tag_col]) 
    #plt.subplot(subplotplace)
    p.div(p.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(10,4))
    #subplotplace += 1
plt.figure(1)
subplotplace = 221
for col in num_col:
    plt.subplot(subplotplace)
    df.groupby(tag_col)[col].mean().plot.bar(figsize = (18,10),title = col.replace('_',' '))
    subplotplace += 1
df[df['Pesticide_Use_Category'] == 2].head()
matrix = df.corr() 
f, ax = plt.subplots(figsize=(25, 12)) 
sns.heatmap(matrix, vmax=.8, square=True, cmap="RdYlGn",annot = True);
df.isnull().sum()
df.loc[(df['Pesticide_Use_Category'] ==1) & (df['Number_Weeks_Used'].isnull()),'Number_Weeks_Used'] = df[df['Pesticide_Use_Category'] ==1]['Number_Weeks_Used'].median()
df.loc[(df['Pesticide_Use_Category'] ==2) & (df['Number_Weeks_Used'].isnull()),'Number_Weeks_Used'] = df[df['Pesticide_Use_Category'] ==2]['Number_Weeks_Used'].median()
df.loc[(df['Pesticide_Use_Category'] ==3) & (df['Number_Weeks_Used'].isnull()),'Number_Weeks_Used'] = df[df['Pesticide_Use_Category'] ==3]['Number_Weeks_Used'].median()
df_test.loc[(df_test['Pesticide_Use_Category'] ==1) & (df_test['Number_Weeks_Used'].isnull()),'Number_Weeks_Used'] = df_test[df_test['Pesticide_Use_Category'] ==1]['Number_Weeks_Used'].median()
df_test.loc[(df_test['Pesticide_Use_Category'] ==2) & (df_test['Number_Weeks_Used'].isnull()),'Number_Weeks_Used'] = df_test[df_test['Pesticide_Use_Category'] ==2]['Number_Weeks_Used'].median()
df_test.loc[(df_test['Pesticide_Use_Category'] ==3) & (df_test['Number_Weeks_Used'].isnull()),'Number_Weeks_Used'] = df_test[df_test['Pesticide_Use_Category'] ==3]['Number_Weeks_Used'].median()
df['total_doses'] = (df['Number_Doses_Week'] * df['Number_Weeks_Used'] )
df_test['total_doses'] = df_test['Number_Doses_Week'] * df_test['Number_Weeks_Used']
df.groupby(tag_col)['total_doses'].mean().plot.bar(figsize = (8,5),title = 'Total Doses')
df_test.isnull().sum()
for col in num_col:
    df[col+"_log"] = np.cbrt(df[col]) 
    df[col+"_log"].hist(bins=20) 
    df_test[col+"_log"] = np.cbrt(df_test[col])
plt.figure(1,figsize = (10,20))
subplotplace = 421
for col in num_col:
    plt.subplot(subplotplace)
    sns.distplot(df[col+"_log"])
    subplotplace += 1
    plt.subplot(subplotplace)
    df[col+"_log"].plot.box()
    subplotplace += 1
for col in num_col:
    df.drop(col,axis = 1, inplace = True)
    
df.head()
for col in num_col:
    df_test.drop(col,axis = 1, inplace = True)
    
df_test.head()
df = df.drop('ID', axis = 1)
df_test = df_test.drop('ID', axis = 1)
X = df.drop('Crop_Damage',1) 
y = df['Crop_Damage']
X=pd.get_dummies(X) 
df=pd.get_dummies(df) 
df_test=pd.get_dummies(df_test)
from sklearn.model_selection import train_test_split
x_train, x_cv, y_train, y_cv = train_test_split(X,y, test_size =0.3)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
model = RandomForestClassifier() 
model.fit(x_train, y_train)
pred_cv = model.predict(x_cv)
accuracy_score(y_cv,pred_cv)
pred_test = model.predict(df_test)
submission=pd.read_csv("../input/av-janatahack-machine-learning-in-agriculture/sample_submission_O1oDc4H.csv")
submission['Crop_Damage']=pred_test 
submission['ID']=df_test_original['ID']
pd.DataFrame(submission, columns=['ID','Crop_Damage']).to_csv('randomforest.csv',index = False)
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
i=1 
kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True) 
for train_index,test_index in kf.split(X,y):     
    print('\n{} of kfold {}'.format(i,kf.n_splits))     
    xtr,xvl = X.loc[train_index],X.loc[test_index]     
    ytr,yvl = y[train_index],y[test_index]         
    model = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=9, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=21,
                       n_jobs=None, oob_score=False, random_state=1, verbose=0,
                       warm_start=False)     
    model.fit(xtr, ytr)     
    pred_test = model.predict(xvl)     
    score = accuracy_score(yvl,pred_test)     
    print('accuracy_score',score)     
    i+=1 

    pred_test = model.predict(df_test)
submission['Crop_Damage']=pred_test 
pd.DataFrame(submission, columns=['ID','Crop_Damage']).to_csv('randomforeststartified.csv',index = False)
importances=pd.Series(model.feature_importances_, index=X.columns) 
importances.plot(kind='barh', figsize=(12,8));
from xgboost import XGBClassifier
i=1 
kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True) 
for train_index,test_index in kf.split(X,y):     
    print('\n{} of kfold {}'.format(i,kf.n_splits))     
    xtr,xvl = X.loc[train_index],X.loc[test_index]     
    ytr,yvl = y[train_index],y[test_index]         
    model = XGBClassifier(n_estimators=50, max_depth=4)     
    model.fit(xtr, ytr)     
    pred_test = model.predict(xvl)     
    score = accuracy_score(yvl,pred_test)     
    print('accuracy_score',score)     
    i+=1 
    pred_test = model.predict(df_test) 
    pred3=model.predict_proba(df_test)[:,1]
submission['Crop_Damage']=pred_test 
pd.DataFrame(submission, columns=['ID','Crop_Damage']).to_csv('xgstartified.csv',index = False)
from sklearn.model_selection import GridSearchCV
# Provide range for max_depth from 1 to 20 with an interval of 2 and from 1 to 200 with an interval of 20 for n_estimators 
paramgrid = {'max_depth': list(range(1, 20, 2)), 'n_estimators': list(range(1, 200, 20))}
grid_search=GridSearchCV(RandomForestClassifier(random_state=1),paramgrid)
x_train, x_cv, y_train, y_cv = train_test_split(X,y, test_size =0.3, random_state=1)
# Fit the grid search model 
grid_search.fit(x_train,y_train)
# Estimating the optimized value 
grid_search.best_estimator_
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(x_train, y_train)
pred_cv = model.predict(x_cv)
accuracy_score(y_cv,pred_cv)
