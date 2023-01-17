import os

import warnings

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sea

import numpy as np

import math as mt

import scipy



from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
url = '../input/cardiovascular-disease/cardiovascular.txt'

data = pd.read_csv(url,sep=';',decimal=',')



# let's separate index from other columns

data.index = data.iloc[:,0]

df = data.iloc[:,1:]



df = df.drop(['chd','famhist'],axis=1)

data.head()

data.shape
df.head()
df.head()
df.dtypes
df = df.astype('float')
df.dtypes
df.describe()
famhist_height = data.famhist.value_counts()/data.shape[0]



chd_height =data.chd.value_counts()/data.shape[0]



print(famhist_height)

print(chd_height)



fig = plt.figure(figsize=(17,4))

ax1 = fig.add_subplot(1, 2, 1)

plt.bar(x=['Absent','Present'],height=famhist_height,color='yellow')

ax2 = fig.add_subplot(1, 2, 2)

plt.bar(x=['No Crises crise','Crise'],height=chd_height,color='b')

plt.show()
plt.figure(figsize=(16,5))

df.boxplot()

plt.title("Distribution of the values ​​of all potential predictors")

plt.show()
df=StandardScaler().fit_transform(df)



df=pd.DataFrame(df,columns=['sbp', 'tobacco', 'ldl', 'adiposity','obesity','alcohol', 'age','typea'])



df.index=data.index
df.head()
df1=pd.concat([df,data.famhist,data.chd],axis=1)



# and take a look 

df1.head()
plt.figure(figsize=(16,5))

sea.boxplot(data=df1.iloc[:,:-2])

plt.title("Distribution of the values ​​of all potential standardized predictors")

plt.grid()

plt.show()
R=round(df.corr(),ndigits=3)

sea.heatmap(round(R,ndigits=2))

plt.show()
n=df.shape[0]

p=df.shape[1]



khi2=-(n-1-(2*p+5)/6)*mt.log(np.linalg.det(R)) # chi test



ddl=p*(p-1)/2



p_valeur=scipy.stats.chi2.pdf(khi2,ddl)
print(p_valeur < 0.01)
inv_R=np.linalg.inv(R)



A=np.zeros(shape=(inv_R.shape[0],inv_R.shape[1]))

for i in range(inv_R.shape[0]):

    for j in range(i+1,inv_R.shape[1]):

        A[i,j]= -inv_R[i,j]/np.sqrt(inv_R[i,i]*inv_R[j,j])

        A[j,i]=A[i,j]

        

R=R.values



kmo_numerateur= np.sum(R**2)-np.sum(np.diag(R**2))

kmo_denominateur=kmo_numerateur + (np.sum(A**2)-np.sum(np.diag(A**2)))

kmo=kmo_numerateur/kmo_denominateur



print("kmo :",round(kmo,ndigits=2))
R=round(df.corr(method='spearman'),ndigits=3)

sea.heatmap(round(R,ndigits=2))

plt.show()
R=round(df.corr(method='kendall'),ndigits=3)

sea.heatmap(round(R,ndigits=2))

plt.show()
#rank the group of records that have the same value

df=df.rank()
from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split

# let's split our data into I/O

X=df.iloc[:,:-1] ; y=df1['chd']



X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=23) 
print( X_train.shape)
print(X_test.shape)
pca=PCA(n_components=5) # split in 5 components



principalComponents = pca.fit_transform(X_train)



factors_Df = pd.DataFrame(data = principalComponents, columns =['PC1','PC2','PC3','PC4','PC5'])



factors_Df.index=X_train.index
# take a look

factors_Df.head()
# let's summary 

print(pca.explained_variance_ratio_)

print(pca.explained_variance_ratio_.cumsum())

print(pca.explained_variance_ratio_.sum())
fig = plt.figure(figsize=(14,6))

ax1 = fig.add_subplot(1, 2, 1)

ax2 = fig.add_subplot(1, 2, 2)



ax1.plot(np.arange(1,6),pca.explained_variance_ratio_,color='b',marker='o')

ax2.plot(np.arange(1,6),np.cumsum(pca.explained_variance_ratio_),color='gold',marker='o')



ax1.set_xlabel('Factors')

ax1.set_title('Explained variance')

ax2.set_xlabel('Factors')

ax2.set_title('Explained variance cumsum')



plt.show()
var_fig=plt.figure(figsize=(9,9)) ; pcs = pca.components_



for i, (x, y) in enumerate(zip(pcs[0, :], pcs[1, :])):

    # show the segment of the origin on point (x, y)

    plt.plot([0, x], [0, y], color='green')

    # show the composition

    plt.text(x, y,df[['sbp','tobacco','ldl','adiposity','obesity','alcohol','age','typea']].columns[i], fontsize='14')



plt.plot([-0.8, 0.8], [0, 0], color='grey', ls='--') # horizontal line y=0    

plt.plot([0, 0], [-0.8, 0.8], color='grey', ls='--') # vertical line x=0

plt.title('Var Projections', fontsize=19)

cercle = plt.Circle((0.515,0.50),0.377,color='k',fill=False)

var_fig.add_artist(cercle)

plt.grid()
names=pd.DataFrame(data=data.chd,columns=['chd'])



final_factors_Df = pd.concat([factors_Df, data['chd']], axis = 1)

id_factors=pd.concat([names,final_factors_Df],axis=1)

id_factors=id_factors.sort_values(by=['PC1'])

final_factors_Df=final_factors_Df.sort_values(by=['PC1'])
fig = plt.figure(figsize = (16,5))

ax = fig.add_subplot(1,1,1)

ax.set_xlabel('Principal Component 1', fontsize = 15)

ax.set_ylabel('Principal Component 2', fontsize = 15)

ax.set_title('Best Plan-1 (Axis 1 & 2)', fontsize = 20)

targets = [0,1]

colors = ['yellow','red']

for target, color in zip(targets,colors):

    indicesToKeep = final_factors_Df['chd'] == target

    ax.scatter(final_factors_Df.loc[indicesToKeep, 'PC1'],final_factors_Df.loc[indicesToKeep, 'PC2'], c = color, s = 50)

    

ax.legend(targets)

ax.grid()

plt.show()
fig = plt.figure(figsize = (16,5))

ax = fig.add_subplot(1,1,1)

ax.set_xlabel('Principal Component 1', fontsize = 15)

ax.set_ylabel('Principal Component 3', fontsize = 15)

ax.set_title('Best Plan-2 (Axis 1 & 3)', fontsize = 20)

targets = [0,1]

colors = ['yellow','red']

for target, color in zip(targets,colors):

    indicesToKeep = final_factors_Df['chd'] == target

    ax.scatter(final_factors_Df.loc[indicesToKeep, 'PC1'], final_factors_Df.loc[indicesToKeep, 'PC3'], c = color, s = 50)

    

ax.legend(targets)

ax.grid()
X_supp=X_test ; y_supp=y_test

coordSupp=pca.transform(X_supp)
coordSupp.shape
fig = plt.figure(figsize = (16,5))

ax = fig.add_subplot(1,1,1)

ax.set_xlabel('Principal Component 1', fontsize = 15)

ax.set_ylabel('Principal Component 2', fontsize = 15)

ax.set_title('Test sample on best Plan-2 (Axis 1 & 2)', fontsize = 20)

df2=pd.DataFrame(data=coordSupp,index=X_supp.index,columns=final_factors_Df.columns[0:5])

df0=df1.loc[X_supp.index]['chd']

gdf=pd.concat([df2,df0],axis=1)

targets = [0,1]

colors = ['green','red']

for target, color in zip(targets,colors):

    indicesToKeep = gdf['chd'] == target

    ax.scatter(gdf.loc[indicesToKeep, 'PC1'], gdf.loc[indicesToKeep, 'PC2'],c= color, s = 50,marker='s')

    ax.legend(targets)



ax.grid()

plt.show()
fig = plt.figure(figsize = (16,5))

ax = fig.add_subplot(1,1,1)

ax.set_xlabel('Principal Component 1', fontsize = 15)

ax.set_ylabel('Principal Component 3', fontsize = 15)

ax.set_title('Test sample on second best Plan-2 (Axis 1 & 3)', fontsize = 20)

df2=pd.DataFrame(data=coordSupp,index=X_supp.index,columns=final_factors_Df.columns[0:5])

df0=df1.loc[X_supp.index]['chd']

gdf=pd.concat([df2,df0],axis=1)

targets = [0,1]

colors = ['green','red']

for target, color in zip(targets,colors):

    indicesToKeep = gdf['chd'] == target

    ax.scatter(gdf.loc[indicesToKeep, 'PC1'], gdf.loc[indicesToKeep, 'PC3'] , c= color, s = 50,marker='s')



ax.legend(targets)

ax.grid()

plt.show()
X_train=pd.concat([data.loc[factors_Df.index]['famhist'],factors_Df],axis=1)

X_train.head()
test_DF=pd.DataFrame(coordSupp,columns=['PC1','PC2','PC3','PC4','PC5'],index=X_test.index)

X_test=pd.concat([data.loc[X_test.index]['famhist'],test_DF],axis=1)



X_test.head()
## import Libraries

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import roc_curve, auc,classification_report

from sklearn.pipeline import Pipeline
pred=[]

X_train=factors_Df.iloc[:,1:4]

param_grid={'bootstrap':[True], 'max_depth':[110,130,150,170],'min_samples_leaf':[7,9,11,13],'min_samples_split':[6,8,12],'n_estimators':[10,15,20,25]}

gs=GridSearchCV(estimator=RandomForestClassifier(),param_grid=param_grid,cv=5)



gs.fit(X_train,y_train)

best_rf=gs.best_estimator_

y_true,y_pred=y_supp,best_rf.predict(X_test.iloc[:,1:4])



df=pd.DataFrame(classification_report(y_true,y_pred,output_dict=True))

sea.heatmap(df.iloc[:-1, :].T, annot=True)
data=data.drop(['typea'],axis=1)

data.head()
X_train=data.loc[X_train.index]; X_test=data.loc[X_test.index]

X_train['famhist'].replace({"Absent":0,"Present":1},inplace=True)

X_test['famhist'].replace({"Absent":0,"Present":1},inplace=True)



y=df1['chd']; y_train=y.loc[y_train.index]; y_test=y[y_test.index]

 



param_grid={'bootstrap':[True], 'max_depth':[110,130,150,170],'min_samples_leaf':[7,9,11,13],'min_samples_split':[8,12,14],'n_estimators':[10,15,20,25]}

gs=GridSearchCV(estimator=RandomForestClassifier(),param_grid=param_grid,cv=5)

gs.fit(X_train,y_train)

best_rf_=gs.best_estimator_

y_true,y_pred=y_supp,best_rf_.predict(X_test)

pred.append(y_pred)
df=pd.DataFrame(classification_report(y_true,y_pred,output_dict=True))

sea.heatmap(df.iloc[:-1, :].T, annot=True)
X_train=factors_Df.iloc[:,1:4]

SVCpipe = Pipeline([('SVC',LinearSVC())])

param_grid = {'SVC__C':np.arange(0.01,100,10)}

linearSVC = GridSearchCV(SVCpipe,param_grid,cv=5,return_train_score=True)

linearSVC.fit(X_train,y_train)

best_svc=linearSVC.best_estimator_

y_true,y_pred=y_supp,best_svc.predict(X_test.iloc[:,1:4])
df=pd.DataFrame(classification_report(y_true,y_pred,output_dict=True))

sea.heatmap(df.iloc[:-1, :].T, annot=True)
X_train=data.loc[X_train.index]; X_test=data.loc[X_test.index]

y=df1['chd']; y_train=y.loc[y_train.index]; y_test=y[y_test.index]

X_train['famhist'].replace({"Absent":0,"Present":1},inplace=True)

X_test['famhist'].replace({"Absent":0,"Present":1},inplace=True)
param_grid = {'SVC__C':np.arange(0.01,100,10)}

linearSVC = GridSearchCV(SVCpipe,param_grid,cv=5,return_train_score=True)

linearSVC.fit(X_train,y_train)

best_svc=linearSVC.best_estimator_

y_true,y_pred=y_supp,best_svc.predict(X_test)

pred.append(y_pred)
df=pd.DataFrame(classification_report(y_true,y_pred,output_dict=True))

sea.heatmap(df.iloc[:-1, :].T, annot=True)
s=pred[0]+pred[1]

pred_finale=[]

for i in range(len(s)):

    if (s[i] >= 1):

        pred_finale.append(1)

    else:

        pred_finale.append(0)

pred_finale=pd.DataFrame(pred_finale)
df=pd.DataFrame(classification_report(y_true,y_pred,output_dict=True))

sea.heatmap(df.iloc[:-1, :].T, annot=True)