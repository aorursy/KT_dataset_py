import pandas as pd

import numpy as np 

import matplotlib.pyplot as plt

import seaborn as sns 

import warnings

warnings.simplefilter("ignore")
data=pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')

data.dropna(axis=0).any()

X=set(data.columns)

X.remove('Outcome')

X=data[X]

y=data['Outcome']

data.head()
print(data.info())

print(data.describe())
age_data=pd.DataFrame(data.groupby(['Age'],as_index=False)['Outcome'].count())

interval={}

temp_sum=0

for i in range(len(age_data)):

    temp_sum+=int(age_data.iloc[i,1])

    if age_data.iloc[i,0]==35:

        interval.update({"20-35":temp_sum})

        temp_sum=0

    elif age_data.iloc[i,0]==50:

        interval.update({"35-50":temp_sum})

        temp_sum=0

    elif age_data.iloc[i,0]==81:

        interval.update({"+50":temp_sum})

plt.figure(figsize=(12,8))

plt.bar(interval.keys(),interval.values(),color=['#cc6699','#339933','#006666'])



plt.xlabel("Age")

plt.ylabel("Count")

plt.title("Counts of Age Interval")

plt.show()
fig, ax = plt.subplots(figsize=(12,7))    

sns.heatmap(data.corr(), annot=True,ax=ax)
sns.pairplot(data, hue="Outcome") 
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

x_scaled=scaler.fit_transform(X)

data_scaled=pd.concat([pd.DataFrame(x_scaled,columns=data.iloc[:,:-1].columns),y],axis=1)
from sklearn.decomposition import PCA

pca=PCA().fit(x_scaled)

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.title("How many variable represents our model with acceptable error PCA")

plt.xlabel("Number of variable")

plt.ylabel("Variance")

plt.grid()
pca=PCA(n_components=5)

x_reduced=pca.fit_transform(data_scaled.iloc[:,:-1])

data_reduced=pd.concat([pd.DataFrame(x_reduced),y],axis=1)

data_reduced.head()
before_x=[]

after_x=[]

data4iqr=data.copy()

for i in range(len(data4iqr.columns)-1):

    col=data4iqr.iloc[:,i:i+1]



    Q1=col.quantile(0.25)

    Q3=col.quantile(0.75)

    IQR=Q3-Q1

    lower=Q1-1.5*IQR

    upper=Q3+1.5*IQR

    new_col=col[~((col<lower)|(col>upper)).any(axis=1)]

    ex_col=col[((col<lower)|(col>upper)).any(axis=1)]

    before_x.append(col)

    data4iqr.drop(index=ex_col.index,axis=0,inplace=True)

    after_x.append(data4iqr.iloc[:,i:i+1])

data4iqr.reset_index(inplace=True)

print("IQR METHOD",len(data)-len(data4iqr)," Row Effected")

####IQR Visualization####

f, axes = plt.subplots(2,5, figsize=(22, 7))

j=0

for i in range(5):

    if data4iqr.columns[i+1]=="Outcome":

        continue

    sns.boxplot(before_x[i],ax=axes[0,j]).set_title(data4iqr.columns[i+1]+" Before IQR")

    sns.boxplot(after_x[i],ax=axes[1,j]).set_title(data4iqr.columns[i+1]+" After IQR")

    j+=1

plt.show()

from sklearn.neighbors import LocalOutlierFactor

clf=LocalOutlierFactor(n_neighbors=20,contamination=0.1)

outlier_pred=clf.fit_predict(data_reduced)

x_score=clf.negative_outlier_factor_

x_score=np.abs(x_score)

xscr_mean=x_score.mean()

xscr_std=np.std(x_score)

lower=xscr_mean-(1*xscr_std)

upper=xscr_mean+(1*xscr_std)
inliers=data[~((x_score>upper)| (x_score<lower))]

print(len(inliers))

lof_data=inliers.copy()
from sklearn import metrics

from sklearn.model_selection import GridSearchCV,cross_val_score
columns=set(lof_data.columns)

columns.remove('Outcome')

x_reduced=lof_data[columns]

y=lof_data['Outcome']

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x_reduced,y,test_size=0.33,random_state=58)
from sklearn.linear_model import LogisticRegression as log_rec

logistic_model=log_rec(C=0.0001,solver='newton-cg')

logistic_model.fit(x_train,y_train)

logistic_pred=logistic_model.predict(x_test)

print("Logistic Regression Accuracy Score Before Tuning %.5f"% metrics.accuracy_score(logistic_pred,y_test))

print("Logistic Regression F1 Score Before Tuning %.5f"%metrics.f1_score(logistic_pred,y_test))
logistic_params={'C':[0.001,0.01,0.1,1,10,100,1000],'solver':[ "liblinear", "sag", "saga","lbfgs"]}

grid=GridSearchCV(log_rec(),logistic_params,scoring='accuracy',cv=3)

grid.fit(x_train,y_train)

grid.best_params_
logistic_model=log_rec(C=grid.best_params_['C'],solver=grid.best_params_['solver'])

logistic_model.fit(x_train,y_train)

logistic_pred=logistic_model.predict(x_test)

print("Logistic Regression Accuracy Score After Tuning %.5f"%metrics.accuracy_score(logistic_pred,y_test))

print("Logistic Regression F1 Score After Tuning %.5f"%metrics.f1_score(logistic_pred,y_test))
from sklearn.neighbors import KNeighborsClassifier

knn_model=KNeighborsClassifier(n_neighbors=3)

knn_model.fit(x_train,y_train)

knn_pred=knn_model.predict(x_test)

print("KNN Accuracy Score Before Tuning %.5f"% metrics.accuracy_score(knn_pred,y_test))

print("KNN F1 Score Before Tuning %.5f"% metrics.f1_score(knn_pred,y_test))
knn_params={'n_neighbors':np.arange(3,90,2)}

grid=GridSearchCV(KNeighborsClassifier(),knn_params,scoring='accuracy',cv=3)

grid.fit(x_train,y_train)

grid.best_params_
knn_model=KNeighborsClassifier(n_neighbors=grid.best_params_['n_neighbors'])

knn_model.fit(x_train,y_train)

knn_pred=knn_model.predict(x_test)

print("KNN Accuracy Score After Tuning %.5f"% metrics.accuracy_score(knn_pred,y_test))

print("KNN F1 Score After Tuning %.5f" % metrics.f1_score(knn_pred,y_test))

#print(knn_model.predict_proba(x_test[3:5]))
from sklearn.tree import DecisionTreeClassifier

dt_model=DecisionTreeClassifier()

dt_model.fit(x_train,y_train)

dt_pred=dt_model.predict(x_test)

print("Decision Tree Accuracy Score Before Tuning %.5f"% metrics.accuracy_score(dt_pred,y_test))

print("Decision Tree F1 Score Before Tuning %.5f"% metrics.f1_score(dt_pred,y_test))
dt_params={'criterion':['gini','entropy'],'max_depth':(2,4,6,8,10,12,16,18,20)}

grid=GridSearchCV(DecisionTreeClassifier(),dt_params,scoring='accuracy')

grid.fit(x_train,y_train)

grid.best_params_
dt_model=DecisionTreeClassifier(criterion=grid.best_params_['criterion'],max_depth=4)

dt_model.fit(x_train,y_train)

dt_pred=dt_model.predict(x_test)

print("Decision Tree Accuracy Score After Tuning %.5f"% metrics.accuracy_score(dt_pred,y_test))

print("Decision Tree F1 Score After Tuning %.5f"% metrics.f1_score(dt_pred,y_test))


from sklearn import tree

plt.figure(figsize=(60,40),dpi=400)

tree.plot_tree(dt_model,filled=True,rounded=True,feature_names=X.columns,

            class_names=['Diabetes','No Diabetes'])

plt.show()

#plt.savefig("tree_visual.png")
from sklearn.svm import SVC

svc_model=SVC()

svc_model.fit(x_train,y_train)

svc_pred=svc_model.predict(x_test)

print("SVM Accuracy Score Before Tuning %.5f"% metrics.accuracy_score(svc_pred,y_test))

print("SVM F1 Score Before Tuning %.5f"%metrics.f1_score(svc_pred,y_test))
svc_params=({'kernel':['rbf'],'C':[0.001,0.1,1,10,100],'gamma':['auto','scale']})

grid=GridSearchCV(SVC(),param_grid=svc_params,scoring="accuracy",cv=3)

grid.fit(x_train,y_train)

grid.best_params_
svc_model=SVC(C=grid.best_params_['C'],kernel=grid.best_params_['kernel'],gamma=grid.best_params_['gamma'])

svc_model.fit(x_train,y_train)

svc_pred=svc_model.predict(x_test)

print("SVM Accuracy Score After Tuning %.5f"% metrics.accuracy_score(svc_pred,y_test))

print("SVM  F1 Score After Tuning %.5f"%metrics.f1_score(svc_pred,y_test))
from sklearn.ensemble import RandomForestClassifier

rf_model=RandomForestClassifier()

rf_model.fit(x_train,y_train)

rf_pred=rf_model.predict(x_test)

print("RandomForest Accuracy Score Before Tuning %.5f"% metrics.accuracy_score(rf_pred,y_test))

print("RandomForest F1 Score Before Tuning %.5f"%metrics.f1_score(rf_pred,y_test))
rf_params={'n_estimators':range(10,110,10),'criterion':['gini','entropy']}

grid=GridSearchCV(RandomForestClassifier(),rf_params,cv=3,scoring='accuracy')

grid.fit(x_train,y_train)

grid.best_params_
rf_model=RandomForestClassifier(n_estimators=grid.best_params_['n_estimators'],criterion=grid.best_params_['criterion'],max_depth=4)

rf_model.fit(x_train,y_train)

rf_pred=rf_model.predict(x_test)

print("RandomForest Accuracy Score After Tuning %.5f"% metrics.accuracy_score(rf_pred,y_test))

print("RandomForest F1 Score After Tuning %.5f"%metrics.f1_score(rf_pred,y_test))
from xgboost import XGBClassifier

xg_model=XGBClassifier()

xg_model.fit(x_train,y_train)

xg_pred=xg_model.predict(x_test)

print("XGBoost Accuracy Score Before Tuning %.5f"% metrics.accuracy_score(xg_pred,y_test))

print("XGBoost F1 Score Before Tuning %.5f"%metrics.f1_score(xg_pred,y_test))
from sklearn.naive_bayes import GaussianNB

nb_model=GaussianNB()

nb_model.fit(x_train,y_train)

nb_pred=nb_model.predict(x_test)

print("NaiveBayes Accuracy Score  %.5f"% metrics.accuracy_score(nb_pred,y_test))

print("NaiveBayes F1 Score  %.5f"%metrics.f1_score(nb_pred,y_test))
from sklearn.cluster import KMeans

km_model=KMeans(n_clusters=2)

km_model.fit(x_train)

km_pred=km_model.predict(x_test)

if metrics.accuracy_score(km_pred,y_test)<0.5:

    zeros=np.where(km_pred==0)

    ones=np.where(km_pred==1)

    km_pred[zeros]=1

    km_pred[ones]=0

metrics.accuracy_score(km_pred,y_test)
km_params={'algorithm':["auto", "full", "elkan"],'max_iter':[100,200,300,400,500,600],'init':['k-means++','random']}

grid=GridSearchCV(KMeans(n_clusters=2,random_state=12),km_params,scoring='accuracy',cv=3)

grid.fit(x_train,y_train)

grid.best_params_
km_model=KMeans(n_clusters=2,init=grid.best_params_['init'],algorithm=grid.best_params_['algorithm'],max_iter=grid.best_params_['max_iter'])

km_model.fit(x_train)

km_pred=km_model.predict(x_test)

if metrics.accuracy_score(km_pred,y_test)<0.5:

    zeros=np.where(km_pred==0)

    ones=np.where(km_pred==1)

    km_pred[zeros]=1

    km_pred[ones]=0

metrics.accuracy_score(km_pred,y_test)
from keras.models import Sequential

from keras.layers import Dense,Flatten

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold
model = Sequential()

model.add(Dense(60, input_dim=x_train.shape[1], activation='relu'))

model.add(Dense(30,activation='relu'))

model.add(Dense(30,activation='relu'))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])

model.fit(x_train,y_train,epochs=50)
eval_score=model.evaluate(x_train, y_train)

print("Loss:",eval_score[0],"Accuracy:",eval_score[1])

from sklearn.ensemble import VotingClassifier 
est=[]

est.append(('svm',svc_model))

est.append(('rf',rf_model))

est.append(('lr',logistic_model))

est.append(('nb',nb_model))

est.append(('xg',xg_model))

vcls=VotingClassifier(estimators=est,voting='hard')

vcls.fit(x_train,y_train)
voting_pred=vcls.predict(x_test)

print("Voting Accuracy Score  %.5f"% metrics.accuracy_score(voting_pred,y_test))

print("Voting F1 Score  %.5f"%metrics.f1_score(voting_pred,y_test))