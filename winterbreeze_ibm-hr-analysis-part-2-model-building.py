import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 

import warnings 
warnings.filterwarnings('ignore')
pd.set_option('display.max_rows',None)
pd.set_option('display.max_column',None)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df = pd.read_csv('/kaggle/input/hrattritioneda/Attrition-EDA.csv')
df.head()
X=df.drop('Attrition',axis=1)
y=df['Attrition']
X.shape
cov_matirx=np.cov(X.T)
eig_vals,eig_vectors=np.linalg.eig(cov_matirx)
eig_vals  # The values are not in order , we need to sort the values 
tot=sum(eig_vals)
var_exp=[(i/tot)*100 for i in sorted(eig_vals,reverse=True)]
cum_var_exp=np.cumsum(var_exp)
print('Cumulative variance Explained:',cum_var_exp)
plt.figure(figsize=(15,4))
plt.bar(range(X.shape[1]),var_exp,alpha=0.5,align='center',label='Individual explained variance')
plt.step(range(X.shape[1]),cum_var_exp,where='mid',label='cummulative explained variance')
plt.ylabel("explained variance ratio")
plt.xlabel("principal components")
plt.legend(loc='best')
plt.tight_layout()
plt.show()
eigen_pairs=[(np.abs(eig_vals[i]),eig_vectors[:,i]) for i in range(len(eig_vals))]

eig_val_sort=[eigen_pairs[index][0] for index in range(len(eig_vals))]
eig_vec_sort=[eigen_pairs[index][1] for index in range(len(eig_vals))]
eig_val_sort.sort(reverse=True)

P_reduce=np.array(eig_vec_sort[0:37]).T
projected_data=np.dot(X,P_reduce)
projected_data_df=pd.DataFrame(projected_data)
projected_data_df.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3, stratify=y)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
print(X.shape)
print(y.shape)
# Applying PCA function on training 
# and testing set of X component 
from sklearn.decomposition import PCA


pca = PCA(n_components = 37)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)                          # LOGISITC REGRESSION WITH PCA

from sklearn.linear_model import LogisticRegression
algo= LogisticRegression(random_state = 3)

algo.fit(X_train_pca , y_train)
y_train_pred = algo.predict(X_train_pca)
y_train_prob = algo.predict_proba(X_train_pca)

#overall acc of train model
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score,roc_curve
from sklearn.model_selection import cross_val_score

print('Confusion matrix - Train :', '\n',confusion_matrix(y_train , y_train_pred))
print('Overall Accuracy - Train :',accuracy_score(y_train , y_train_pred))
print('AUC - Train:', roc_auc_score(y_train , y_train_prob[:,1]))

y_test_pred = algo.predict(X_test_pca)
y_test_prob = algo.predict_proba(X_test_pca)[:,1]
print('*'*50)
print('Confusion matrix - Test :', '\n',confusion_matrix(y_test , y_test_pred))
print('Overall Accuracy - Test :',accuracy_score(y_test , y_test_pred))
print('AUC - Test:', roc_auc_score(y_test , y_test_prob))

print('*'*50)
scores=cross_val_score(algo,X,y,cv=3,scoring='roc_auc')
print('Cross Val Scores')
print(scores)
print('Bias Error    :',100-scores.mean()*100)
print('Variance Error:',scores.std()*100)



fpr , tpr , threshold = roc_curve(y_test , y_test_prob)
plt.plot(fpr , tpr)
plt.plot(fpr , fpr , 'r-')
plt.xlabel('FPR')
plt.ylabel('TPR')

from sklearn.metrics import confusion_matrix , accuracy_score , roc_auc_score , roc_curve
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB,BernoulliNB,MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score,KFold

# Importing all the predictive models.


lr = LogisticRegression(fit_intercept=True)
bagged_lr = BaggingClassifier(base_estimator = lr, n_estimators = 25, random_state = 3)
gnb= GaussianNB()
bnb= BernoulliNB()
mnb= MultinomialNB()
knn = KNeighborsClassifier()
dtc = DecisionTreeClassifier(ccp_alpha=0.01) # to increase pruning and avoid overfitting
rfc= RandomForestClassifier()
svm= SVC(probability=True)

# Declaring various classification models for the predictive model building.
clf=DummyClassifier(strategy='stratified')
clf.fit(X_train,y_train)
clf.predict(X_test)
print('Base Score on Train Data Set: ',clf.score(X_train,y_train)) 
print('Base Score on Test Data Set : ',clf.score(X_test,y_test)) 
from sklearn.metrics import classification_report
def model_eval(algo , X_train , y_train , X_test , y_test):

    algo.fit(X_train , y_train)
    y_pred = algo.predict(X_train)

    y_train_pred = algo.predict(X_train)               # Finding the positives and negatives 
    y_train_prob = algo.predict_proba(X_train)[:,1]    #we are intersted only in the second column


    #overall acc of train model
    print('Confusion matrix - Train :', '\n',confusion_matrix(y_train , y_train_pred))
    print('Overall Accuracy - Train :',accuracy_score(y_train , y_train_pred))
    print('AUC - Train:', roc_auc_score(y_train , y_train_prob))

    y_test_pred = algo.predict(X_test)
    y_test_prob = algo.predict_proba(X_test)[:,1]
    print('*'*50)
    print('Confusion matrix - Test :', '\n',confusion_matrix(y_test , y_test_pred))
    print('Overall Accuracy - Test :',accuracy_score(y_test , y_test_pred))
    print('AUC - Test:', roc_auc_score(y_test , y_test_prob))
    
    print('*'*50)
    scores=cross_val_score(algo,X,y,cv=3,scoring='roc_auc')
    print('Cross Val Scores')
    print(scores)
    print('Bias Error    :',100-scores.mean()*100)
    print('Variance Error:',scores.std()*100)
    
    print('\n')
    print('Classification Report:\n', classification_report(y_test, y_test_pred))
    
    

    fpr , tpr , threshold = roc_curve(y_test , y_test_prob)
    plt.plot(fpr , tpr)
    plt.plot(fpr , fpr , 'r-')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
model_eval(bagged_lr , X_train , y_train , X_test , y_test)
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.model_selection import cross_val_score, cross_val_predict
bagged_lr=BaggingClassifier(base_estimator=lr,n_estimators=15,random_state=3)
adaboost_lr=AdaBoostClassifier(base_estimator=lr,n_estimators=50,random_state=3)   #default decision tree
gb=GradientBoostingClassifier(n_estimators=55,random_state=3)                   # Cannot have base_estimator

models=[]
models.append(('Bagged_Logisitc_Regression',bagged_lr))
models.append(('Ada_Boost_Logistic_Regression',adaboost_lr))
models.append(('Gradient_Boost',gb))



results=[]
names=[]
for name,model in models:
    kfold=KFold(n_splits=5,shuffle=True,random_state=0)
    cv_result=cross_val_score(model,X_train,y_train,cv=kfold,scoring='roc_auc')
    results.append(cv_result)
    names.append(name)
    print("%s: %f (%f)" % (name,np.mean(cv_result),np.var(cv_result,ddof=1)))
fig=plt.figure(figsize=(15,8))
fig.suptitle("Algorithm comparision")
ax=fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names,fontsize=8)
plt.show()
model_eval(knn , X_train , y_train , X_test , y_test)
bagged_knn=BaggingClassifier(base_estimator=knn,n_estimators=15,random_state=3) # default DT, cannot use RandomForest
adaboost=AdaBoostClassifier(n_estimators=50,random_state=3)                    # default decision tree, cannot use KNN


models=[]
models.append(('Bagged_KNN',bagged_knn))
models.append(('Ada_Boost',adaboost))




results=[]
names=[]
for name,model in models:
    kfold=KFold(n_splits=5,shuffle=True,random_state=0)
    cv_result=cross_val_score(model,X_train,y_train,cv=kfold,scoring='roc_auc')
    results.append(cv_result)
    names.append(name)
    print("%s: %f (%f)" % (name,np.mean(cv_result),np.var(cv_result,ddof=1)))
fig=plt.figure(figsize=(15,8))
fig.suptitle("Algorithm comparision")
ax=fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names,fontsize=8)
plt.show()
model_eval(gnb , X_train , y_train , X_test , y_test)
gnb= GaussianNB()
bnb= BernoulliNB()

gaussian_bag=BaggingClassifier(base_estimator=gnb,n_estimators=10,random_state=3)
gaussian_adaboost=AdaBoostClassifier(base_estimator=gnb,n_estimators=30,random_state=3)
bernoulli_bag=BaggingClassifier(base_estimator=bnb,n_estimators=10,random_state=3)
bernoulli_adaboost=AdaBoostClassifier(base_estimator=bnb,n_estimators=30,random_state=3)



models=[]
models.append(('Naive_Bayes_Gaussian',gnb))
models.append(('Naive_Bayes_Bernoulli',bnb))
models.append(('Gaussian_bagged',gaussian_bag))
models.append(('Bernoulli_bagged',bernoulli_bag))
models.append(('Adaboost_Gaussian',gaussian_adaboost))
models.append(('Adaboost_Bernoulli',bernoulli_adaboost))



results=[]
names=[]
for name,model in models:
    kfold=KFold(n_splits=5,shuffle=True,random_state=0)
    cv_result=cross_val_score(model,X_train,y_train,cv=kfold,scoring='roc_auc')
    results.append(cv_result)
    names.append(name)
    print("%s: %f (%f)" % (name,np.mean(cv_result),np.var(cv_result,ddof=1)))
fig=plt.figure(figsize=(15,8))
fig.suptitle("Algorithm comparision")
ax=fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names,fontsize=8)
plt.show()
model_eval(dtc , X_train , y_train , X_test , y_test)
bagged_dtc=BaggingClassifier(n_estimators=15,random_state=3)       # default decision tree, cannot use RandomF 
adaboost_dtc=AdaBoostClassifier(n_estimators=50,random_state=3)       # default decision tree, cannot use KNN


models=[]
models.append(('Bagged_DTC',bagged_dtc))
models.append(('Adaboost_DTC',adaboost_dtc))




results=[]
names=[]
for name,model in models:
    kfold=KFold(n_splits=5,shuffle=True,random_state=0)
    cv_result=cross_val_score(model,X_train,y_train,cv=kfold,scoring='roc_auc')
    results.append(cv_result)
    names.append(name)
    print("%s: %f (%f)" % (name,np.mean(cv_result),np.var(cv_result,ddof=1)))
fig=plt.figure(figsize=(15,8))
fig.suptitle("Algorithm comparision")
ax=fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names,fontsize=8)
plt.show()
model_eval(rfc , X_train , y_train , X_test , y_test)
bagged_rfc=BaggingClassifier(base_estimator=rfc,n_estimators=15,random_state=3)       # default decision tree, cannot use RandomF 
adaboost_rfc=AdaBoostClassifier(base_estimator=rfc,n_estimators=50,random_state=3)   # default decision tree, cannot use KNN
             

models=[]
models.append(('Bagged_RFC',bagged_rfc))
models.append(('Adaboost_RFC',adaboost_rfc))




results=[]
names=[]
for name,model in models:
    kfold=KFold(n_splits=5,shuffle=True,random_state=0)
    cv_result=cross_val_score(model,X_train,y_train,cv=kfold,scoring='roc_auc')
    results.append(cv_result)
    names.append(name)
    print("%s: %f (%f)" % (name,np.mean(cv_result),np.var(cv_result,ddof=1)))
fig=plt.figure(figsize=(15,8))
fig.suptitle("Algorithm comparision")
ax=fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names,fontsize=8)
plt.show()
from scipy.stats import randint as sp_randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

rfc = RandomForestClassifier(random_state=3)
params = { 'n_estimators' : sp_randint(50 , 200) , 
           'max_features' : sp_randint(1,26) ,
           'max_depth' : sp_randint(2,10) , 
           'min_samples_split' : sp_randint(2,10) ,
           'min_samples_leaf' : sp_randint(1,10) ,
           'criterion' : ['gini' , 'entropy']
    
}

rsearch_rfc = RandomizedSearchCV(rfc , param_distributions= params , n_iter= 200 , cv = 3 , scoring='roc_auc' , random_state= 3 , return_train_score=True , n_jobs=-1)

rsearch_rfc.fit(X,y)
rsearch_rfc.best_params_    
rfc= RandomForestClassifier(**rsearch_rfc.best_params_,random_state=3)

rfc.fit(X_train,y_train)

y_train_pred=rfc.predict(X_train)                 # Finding the Positives and Negatives 
y_train_prob=rfc.predict_proba(X_train)[:,1]      # We are interested only in the 2nd column



print('Confusion Matrix - Train:','\n' ,confusion_matrix(y_train,y_train_pred))
print('Overall Accuracy - Train:', accuracy_score(y_train,y_train_pred))             #Train
print('AUC- Train',roc_auc_score(y_train,y_train_prob))

y_test_pred=rfc.predict(X_test)
y_test_prob=rfc.predict_proba(X_test)[:,1]


print('\n')
print('Confusion Matrix - Test:','\n' ,confusion_matrix(y_test,y_test_pred))
print('Overall Accuracy - Test:', accuracy_score(y_test,y_test_pred))               #Test
print('AUC- Test',roc_auc_score(y_test,y_test_prob))

print('\n')
fpr,tpr,thresholds= roc_curve(y_test,y_test_prob)
plt.plot(fpr,tpr)
plt.plot(fpr,fpr,'r-')
plt.xlabel('FPR')
plt.ylabel('TPR')
col_sorted_by_importance=rfc.feature_importances_.argsort()
feat_imp=pd.DataFrame({
    'cols':X.columns[col_sorted_by_importance],
    'imps':rfc.feature_importances_[col_sorted_by_importance]
})

feat_imp.sort_values(by='imps',ascending=False)[:10]
model_eval(svm, X_train , y_train , X_test , y_test)
from sklearn.svm import SVC
svm=SVC(probability=True)

kernel=['linear','poly','rbf','sigmoid']

for i in kernel:
    svm=SVC(kernel=i,C=1.0)
    svm.fit(X_train,y_train)
    print('For kernel i,',i)
    print('accuracy is' ,svm.score(X_test,y_test))
from sklearn.model_selection import GridSearchCV 
svm=SVC(probability=True,class_weight='balanced',random_state=3)
param_grid = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'coef0':[0.001,10,0.5],
              'kernel': ['rbf','poly', 'sigmoid']}  
  
grid_search_svm = GridSearchCV(svm, param_grid, refit = True, verbose = 3) 
  
# fitting the model for grid search 
grid_search_svm.fit(X_train,y_train) 
# print best parameter after tuning 
print(grid_search_svm.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid_search_svm.best_estimator_) 
svm= SVC(probability=True,**grid_search_svm.best_params_,random_state=3)

svm.fit(X_train,y_train)

y_train_pred=svm.predict(X_train)                 # Finding the Positives and Negatives 
y_train_prob=svm.predict_proba(X_train)[:,1]      # We are interested only in the 2nd column



print('Confusion Matrix - Train:','\n' ,confusion_matrix(y_train,y_train_pred))
print('Overall Accuracy - Train:', accuracy_score(y_train,y_train_pred))             #Train
print('AUC- Train',roc_auc_score(y_train,y_train_prob))

y_test_pred=svm.predict(X_test)
y_test_prob=svm.predict_proba(X_test)[:,1]


print('\n')
print('Confusion Matrix - Test:','\n' ,confusion_matrix(y_test,y_test_pred))
print('Overall Accuracy - Test:', accuracy_score(y_test,y_test_pred))               #Test
print('AUC- Test',roc_auc_score(y_test,y_test_prob))

print('\n')
fpr,tpr,thresholds= roc_curve(y_test,y_test_prob)
plt.plot(fpr,tpr)
plt.plot(fpr,fpr,'r-')
plt.xlabel('FPR')
plt.ylabel('TPR')
bagged_svm=BaggingClassifier(base_estimator=svm,n_estimators=15,random_state=3)       # default decision tree, cannot use RandomF 
adaboost_svm=AdaBoostClassifier(base_estimator=svm,n_estimators=15,random_state=3)   # default decision tree, cannot use KNN
gb_lr=GradientBoostingClassifier(n_estimators=55,random_state=3)                # Does not have base_estimator, uses DT as stump

models=[]
models.append(('Bagged_SVM',bagged_svm))
models.append(('Adaboost_SVM',adaboost_svm))
models.append(('Gradient_Boost',gb_lr))



results=[]
names=[]
for name,model in models:
    kfold=KFold(n_splits=5,shuffle=True,random_state=0)
    cv_result=cross_val_score(model,X_train,y_train,cv=kfold,scoring='roc_auc')
    results.append(cv_result)
    names.append(name)
    print("%s: %f (%f)" % (name,np.mean(cv_result),np.var(cv_result,ddof=1)))
fig=plt.figure(figsize=(15,8))
fig.suptitle("Algorithm comparision")
ax=fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names,fontsize=8)
plt.show()
Xytrain=pd.concat([X_train,y_train],axis=1)

print('Before Oversampling:','\n',Xytrain['Attrition'].value_counts())

Xytrain0=Xytrain[Xytrain['Attrition']==0]
Xytrain1=Xytrain[Xytrain['Attrition']==1]

len0=len(Xytrain0)
len1=len(Xytrain1)

Xytrain1_os=Xytrain1.sample(len0,replace=True,random_state=3) # To duplicate the values when over sampling [replace=True]
Xytrain_os=pd.concat([Xytrain0,Xytrain1_os],axis=0)           # Axis 0 because it is appending and not merging 

print('\n')
print('After Oversampling:','\n',Xytrain_os['Attrition'].value_counts())
X_os=Xytrain_os.drop('Attrition',axis=1)
y_os=Xytrain_os['Attrition']
from sklearn.model_selection import train_test_split
X_train_os, X_test_os, y_train_os, y_test_os = train_test_split(X_os, y_os, test_size=0.3, random_state=3)
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
X_train_os_scaled = ss.fit_transform(X_train_os)
X_test_os_scaled = ss.transform(X_test_os)
svm_os= SVC(probability=True)
model_eval(svm_os, X_train_os_scaled , y_train_os , X_test_os_scaled , y_test_os)
from imblearn.over_sampling import SMOTE,SVMSMOTE

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30,random_state = 7)

smote=SVMSMOTE(sampling_strategy='minority',random_state=3)
X_train_sm,y_train_sm=smote.fit_sample(X_train,y_train)
#smote = SMOTE(sampling_strategy = 'minority', random_state = 3)
#X_train_sm, y_train_sm = smote.fit_sample(X_train, y_train)
svm= SVC(probability=True)

svm.fit(X_train_sm,y_train_sm)

y_train_pred=svm.predict(X_train_sm)                 # Finding the Positives and Negatives 
y_train_prob=svm.predict_proba(X_train_sm)[:,1]      # We are interested only in the 2nd column



print('Confusion Matrix - Train:','\n' ,confusion_matrix(y_train_sm,y_train_pred))
print('Overall Accuracy - Train:', accuracy_score(y_train_sm,y_train_pred))             #Train
print('AUC- Train',roc_auc_score(y_train_sm,y_train_prob))

y_test_pred=svm.predict(X_test)
y_test_prob=svm.predict_proba(X_test)[:,1]


print('\n')
print('Confusion Matrix - Test:','\n' ,confusion_matrix(y_test,y_test_pred))
print('Overall Accuracy - Test:', accuracy_score(y_test,y_test_pred))               #Test
print('AUC- Test',roc_auc_score(y_test,y_test_prob))

print('\n')
fpr,tpr,thresholds= roc_curve(y_test,y_test_prob)
plt.plot(fpr,tpr)
plt.plot(fpr,fpr,'r-')
plt.xlabel('FPR')
plt.ylabel('TPR')
from imblearn.over_sampling import ADASYN 
adasyn = ADASYN(sampling_strategy='auto')
X_train_adasyn,y_train_adasyn=adasyn.fit_sample(X_train,y_train)
y_train_adasyn.value_counts()
model_eval(svm, X_train_adasyn , y_train_adasyn , X_test , y_test)
import lightgbm as lgb
lgbm = lgb.LGBMClassifier()

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform

params = {
    'n_estimators' : sp_randint(50,200) , 
    'max_depth' : sp_randint(2,15) ,
    'learning_rate' : sp_uniform(0.001 , 0.5 ) ,
    'num_leaves' : sp_randint(20 , 50) 
} 


rsearch = RandomizedSearchCV(lgbm , param_distributions= params , cv = 3 , n_iter= 200 , n_jobs=-1 ,random_state= 3)

rsearch.fit(X , y)
rsearch.best_estimator_
lgbm= lgb.LGBMClassifier(**rsearch.best_params_)

lgbm.fit(X_train,y_train)

y_train_pred=lgbm.predict(X_train)                 # Finding the Positives and Negatives 
y_train_prob=lgbm.predict_proba(X_train)[:,1]      # We are interested only in the 2nd column

print('Confusion Matrix - Train:','\n' ,confusion_matrix(y_train,y_train_pred))
print('Overall Accuracy - Train:', accuracy_score(y_train,y_train_pred))             #Train
print('AUC- Train',roc_auc_score(y_train,y_train_prob))

y_test_pred=lgbm.predict(X_test)
y_test_prob=lgbm.predict_proba(X_test)[:,1]


print('\n')
print('Confusion Matrix - Test:','\n' ,confusion_matrix(y_test,y_test_pred))
print('Overall Accuracy - Test:', accuracy_score(y_test,y_test_pred))               #Test
print('AUC- Test',roc_auc_score(y_test,y_test_prob))

print('\n')
fpr,tpr,thresholds= roc_curve(y_test,y_test_prob)
plt.plot(fpr,tpr)
plt.plot(fpr,fpr,'r-')
plt.xlabel('FPR')
plt.ylabel('TPR')
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

params = {
        'min_child_weight': [1,2,3,4,5,6,7,8,9,10],
        'gamma': [0.5, 1,1,1.25,1.35,1.45, 1.5,1.75, 2, 5],
        'subsample': [0.6,0.7 ,0.8,0.9, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0,1.1,1.2],
        'max_depth': [3, 4, 5,6,7]
        }

xgb = XGBClassifier(learning_rate=0.02, n_estimators=1000, objective='binary:logistic',
                    silent=True, nthread=1)

folds = 3
param_comb = 5

skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 3)

random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb, scoring='accuracy', n_jobs=4, cv=skf.split(X,y), verbose=3, random_state=3 )

random_search.fit(X, y)
xgb= XGBClassifier(**random_search.best_params_,random_state=3)

xgb.fit(X_train,y_train)

y_train_pred=xgb.predict(X_train)                 # Finding the Positives and Negatives 
y_train_prob=xgb.predict_proba(X_train)[:,1]      # We are interested only in the 2nd column



print('Confusion Matrix - Train:','\n' ,confusion_matrix(y_train,y_train_pred))
print('Overall Accuracy - Train:', accuracy_score(y_train,y_train_pred))             #Train
print('AUC- Train',roc_auc_score(y_train,y_train_prob))

y_test_pred=xgb.predict(X_test)
y_test_prob=xgb.predict_proba(X_test)[:,1]


print('\n')
print('Confusion Matrix - Test:','\n' ,confusion_matrix(y_test,y_test_pred))
print('Overall Accuracy - Test:', accuracy_score(y_test,y_test_pred))               #Test
print('AUC- Test',roc_auc_score(y_test,y_test_prob))

print('\n')
fpr,tpr,thresholds= roc_curve(y_test,y_test_prob)
plt.plot(fpr,tpr)
plt.plot(fpr,fpr,'r-')
plt.xlabel('FPR')
plt.ylabel('TPR')
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
svm=SVC(probability=True)
stacked=VotingClassifier(estimators=[('Bagged_Logistic_Regression',bagged_lr),('Adaboost_Bernoulli_Naive_Bayes',bernoulli_adaboost),('GBOOst',gb),('Bagged_RandomForest',bagged_rfc),('Support_Vector_Machines',svm)],voting='soft')
models=[]
models.append(('Bagged_Logistic_Regression',bagged_lr))
models.append(('Adaboost_Bernoulli_Naive_Bayes',bernoulli_adaboost))
models.append(('GBOOst',gb))
models.append(('Bagged_RandomForest',bagged_rfc))
models.append(('Support_Vector_Machines',svm))
models.append(('Stacked',stacked))


results=[]
names=[]
for name,model in models:
    kfold=KFold(n_splits=5,shuffle=True,random_state=0)
    cv_result=cross_val_score(model,X,y,cv=kfold,scoring='roc_auc')
    results.append(cv_result)
    names.append(name)
    print("%s: %f (%f)" % (name,np.mean(cv_result),np.var(cv_result,ddof=1)))
fig=plt.figure(figsize=(15,8))
fig.suptitle("Algorithm comparision")
ax=fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
stacked11=VotingClassifier(estimators=[('Bagged_Logistic_Regression',bagged_lr),('GBOOst',gb),('Support_Vector_Machines',svm)],voting='soft')
models=[]
models.append(('Bagged_Logistic_Regression',bagged_lr))
models.append(('GBOOst',gb))
models.append(('Support_Vector_Machines',svm))
models.append(('Stacked',stacked))


results=[]
names=[]
for name,model in models:
    kfold=KFold(n_splits=5,shuffle=True,random_state=0)
    cv_result=cross_val_score(model,X,y,cv=kfold,scoring='roc_auc')
    results.append(cv_result)
    names.append(name)
    print("%s: %f (%f)" % (name,np.mean(cv_result),np.var(cv_result,ddof=1)))
fig=plt.figure(figsize=(15,8))
fig.suptitle("Algorithm comparision")
ax=fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
data=pd.read_csv('/kaggle/input/hrattritioneda/Attrition-EDA.csv')
data.head()
from pycaret.classification import *
clf=setup(data,target='Attrition')
compare_models()
from pycaret.classification import *
clf=setup(data,target='Attrition',normalize=True,
    normalize_method='zscore',
    transformation=True,
    transformation_method='yeo-johnson')
compare_models(sort='AUC')
lr.classes_ = np.array([-1, 1])
tuned_lr= tune_model(lr,optimize='AUC')
evaluate_model(tuned_lr)
final_lr_model=finalize_model(tuned_lr)
print(final_lr_model)
predictions=predict_model(final_lr_model,data=data)
predictions.head()
columns=[column for column in predictions.columns if (column!='Attrition') & (column!='Label') &(column!='Score')]
columns= columns + ['Attrition','Label','Score']
predictions=predictions[columns]
predictions['Attrition'].unique()
predictions['Label'].unique()
predictions.head()
from PIL import Image
Image.open('/kaggle/input/employee-retention/employee-retention-strategy-slide13.png')