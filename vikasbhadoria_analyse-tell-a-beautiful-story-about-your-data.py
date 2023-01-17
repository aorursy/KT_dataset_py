import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import plotly.express as px
warnings.filterwarnings("ignore")
plt.style.use('ggplot')
df = pd.read_csv('../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
df.head()
df.salary.max()
df[df['salary'] == 940000.0]
# Let's replace 'Workex' , 'status' columns with integers before performing any visualizations i am planning to keep gender same as of now for visualizations.
df['status'] = df['status'].apply(lambda x:1 if x == 'Placed' else 0)
df['workex'] = df['workex'].apply(lambda x:1 if x == 'Yes' else 0)
sns.heatmap(df.isnull(),cmap='Blues',cbar=False, yticklabels=False)
placed_df = df[df['status']==1]
not_placed = df[df['status']==0]
df.hist(bins=30,color='g',figsize=(14,10),ec="black")
var = 'salary'
fig, ax = plt.subplots()
fig.set_size_inches(20, 8)
plt.xticks(rotation=90);
sns.countplot(x = var,palette="cool", data = placed_df)
ax.set_xlabel('Salary', fontsize=16)
ax.set_ylabel('Count', fontsize=16)
ax.set_title('Salary Count Distribution', fontsize=15)
sns.despine()
plt.figure(figsize=(12, 8))
sns.set(font_scale=1)
correlations = df.corr()
sns.heatmap(correlations,cmap="YlGnBu",linewidths=.5, annot=True)

plt.xticks(fontsize=14, rotation = 90)
plt.yticks(fontsize=14, rotation = 0)
plt.title('Correlations between numerical values of the dataset', fontsize=20)
plt.show()
df_to_plot = df[['gender','ssc_b','hsc_b', 'hsc_s', 'degree_t', 'specialisation','workex']]
for i in df_to_plot.columns:
    plt.figure(figsize=(18,8))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel(xlabel=i, fontsize=16)
    plt.ylabel(ylabel='count', fontsize=16)
    sns.countplot(x = i, hue = 'status', data = df, palette="vlag")
plt.figure(figsize = (14,6))
sns.kdeplot(not_placed['degree_p'], label = 'Students not placed', color = 'r', shade = True)
sns.kdeplot(placed_df['degree_p'],label='Students who got placed',color = 'b',shade=True)
plt.xlabel('Percentage obtained in degree')
plt.figure(figsize = (14,6))
sns.kdeplot(not_placed['ssc_p'], label = 'Students not placed', color = 'r', shade = True)
sns.kdeplot(placed_df['ssc_p'],label='Students who got placed',color = 'b',shade=True)
plt.xlabel('Marks obtained in ssc_b')
plt.figure(figsize = (14,6))
sns.kdeplot(not_placed['hsc_p'], label = 'Students not placed', color = 'r', shade = True)
sns.kdeplot(placed_df['hsc_p'],label='Students who got placed',color = 'b',shade=True)
plt.xlabel('Marks obtained in hsc_p')
plt.figure(figsize = (14,6))
sns.kdeplot(not_placed['etest_p'], label = 'Students not placed', color = 'r', shade = True)
sns.kdeplot(placed_df['etest_p'],label='Students who got placed',color = 'b',shade=True)
plt.xlabel('Marks obtained in etest')
percentage_corr = df[['ssc_p','hsc_p','degree_p','mba_p','etest_p']]
percentage_corr_df = percentage_corr.corr()
sns.heatmap(percentage_corr_df,cmap="Blues",linewidths=.5, annot=True)
plt.figure(figsize=(14,6))
ax = sns.boxplot(x="specialisation", y="mba_p", hue="gender",
                 data=df, palette="Set3")
plt.figure(figsize=(14,4))
sns.boxplot(x='salary',y='gender',data=placed_df)
plt.figure(figsize=(14,4))
sns.boxplot(x='salary',y='specialisation',data=placed_df)
plt.figure(figsize=(14,4))
sns.boxplot(x='salary',y='degree_t',data=placed_df)
fig = px.scatter(placed_df,x='mba_p', y='salary')
fig.update_layout(title='Salary v/s Percentage in MBA',xaxis_title="MBA_Per",yaxis_title="Salary")
fig.show()
fig = px.scatter(placed_df,x='etest_p', y='salary')
fig.update_layout(title='Salary v/s etest_p',xaxis_title="etest_p",yaxis_title="Salary")
fig.show()
grdsp = df.groupby(["degree_t"])[["degree_p"]].mean().reset_index()

fig = px.pie(grdsp,
             values="degree_p",
             names="degree_t",
             template="seaborn")
fig.update_traces(rotation=45, pull=0.01, textinfo="percent+label")
fig.show()
grdsp = df.groupby(["specialisation"])[["mba_p"]].mean().reset_index()

fig = px.pie(grdsp,
             values="mba_p",
             names="specialisation",
             template="seaborn")
fig.update_traces(rotation=45, pull=0.01, textinfo="percent+label")
fig.show()
grdsp = df.groupby(["hsc_s"])[["hsc_p"]].mean().reset_index()

fig = px.pie(grdsp,
             values="hsc_p",
             names="hsc_s",
             template="seaborn")
fig.update_traces(rotation=45, pull=0.01, textinfo="percent+label")
fig.show()
df_ML_status = df.drop(['sl_no'],axis=1)
df_ML_status['gender'] = df_ML_status['gender'].apply(lambda x:1 if x == 'M' else 0)
df_ML_status.head()
df_ML_status = pd.get_dummies(df_ML_status, drop_first=True)
df_ML_status.head()
plt.figure(figsize=(15,15))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(df_ML_status.corr(),vmax=.3, center=0, cmap=cmap,annot=True,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
df_ML_status = df_ML_status.drop(['salary'],axis=1)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
df_ML_status = df_ML_status.drop('status',axis=1)
y = df['status']
X_train,X_test,y_train,y_test=train_test_split(df_ML_status,y,test_size=0.2,random_state=42)
X_train.shape,X_test.shape
LR_model=LogisticRegression()
LR_model.fit(X_train,y_train)
predict=LR_model.predict(X_test)
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
print(accuracy_score(predict,y_test))
from sklearn.ensemble import RandomForestClassifier
rf_model=RandomForestClassifier()
rf_model.fit(X_train,y_train)
predict_rf = rf_model.predict(X_test)
print(accuracy_score(predict_rf,y_test))
import xgboost
classifier=xgboost.XGBClassifier()
classifier.fit(X_train,y_train)
predict_xg = classifier.predict(X_test)
print(accuracy_score(predict_xg,y_test))
from sklearn.model_selection import RandomizedSearchCV
param_grid_LR = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],'class_weight':['balanced', None],'max_iter':[100,200,300,400,1000,1500] }
random_search_LR=RandomizedSearchCV(LR_model,param_distributions=param_grid_LR
                                    ,n_iter=10,scoring='roc_auc',n_jobs=-1,cv=5,verbose=3)
random_search_LR.fit(df_ML_status,y)
random_search_LR.best_params_
random_search_LR.best_score_