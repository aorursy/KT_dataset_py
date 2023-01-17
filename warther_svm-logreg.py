import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

%matplotlib inline



way='../input'



main_df=pd.read_csv(way+'/diabetes.csv')



print(main_df.head())

##### Split between X and y #####



X_main=main_df.drop('Outcome',axis=1)

y_main=main_df['Outcome']



##### Split between train and main #####



from sklearn.model_selection import train_test_split



X_train,X_test,y_train,y_test=train_test_split(X_main,y_main,

                                               test_size=0.33,random_state=1)



print('train size is %i'%y_train.shape[0])

print('test size is %i'%y_test.shape[0])
names=main_df.columns.values



##### Repérer les valeurs abstraites #####

         

zeros=[]

for i in range(len(names)):

    zeros.append(np.count_nonzero(main_df[names[i]]==0))

CountZero=pd.DataFrame({'names':names,'zeros':zeros})

print(CountZero)



##### Transform the O in -1, except for pregnancies #####



for j in range(1,len(names)-1):

    for i in range(len(main_df)):

        if main_df.iloc[i,j]==0:

            main_df.iloc[i,j]=-1
##### Covariance matrix #####



corr=X_main.corr()

plt.matshow(corr)



y_label=X_main.columns.values

y_pos=np.arange(len(y_label))

x_label=y_label.copy()

for i in range(len(x_label)):

    x_label[i]=y_label[i][:4]

x_pos=np.arange(len(x_label))

plt.xticks(x_pos,x_label)

plt.yticks(y_pos,y_label)

plt.colorbar()

plt.title('Covariance Matrix')



plt.show()
##### Histogrammes ###### 



plt.figure(2)

plt.hist(main_df['Pregnancies'],color='g',bins=range(0,20),align='left')

plt.title('Histogramme Pregnancies')

plt.show()



plt.figure(3)

plt.hist(main_df['Age'],color='y',bins=range(20,90,2),align='left')

plt.title('Histogramme Age')

plt.show()



plt.figure(4)

plt.hist2d(main_df['Age'],main_df['Pregnancies'],bins=[range(21,80,2),range(0,20)])

plt.xlabel('Age')

plt.ylabel('Pregnancies')

plt.title('Histogramme 2D Pregnancies/Age')

plt.show()
from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression



from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import PolynomialFeatures

from sklearn.preprocessing import Imputer



from sklearn.model_selection import GridSearchCV



from sklearn import metrics



from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score
## classifier

svm=SVC(probability=True)

reglog=LogisticRegression()



## preprocess

scale_pipe=StandardScaler()

poly=PolynomialFeatures(degree=2)

imput=Imputer(missing_values=-1,strategy='mean')



## grid parameters

param_svm=dict(clf__C=[0.001,0.1,1,10],clf__kernel=['rbf','linear','sigmoid'])

param_reglog=dict(clf__C=[0.001,0.1,1,10])



## declaration

clf_name=['SVM','RegLog']

clf=[svm,reglog]

param_grid=[param_svm,param_reglog]



auc_all=[]

fpr_all=[]

tpr_all=[]



train_size_all=[]

train_score_all=[]

cv_score_all=[]


for i in [0,1]:



### pipeline ###



    pipe=Pipeline([('imput',imput),('poly',poly),

                   ('scale',scale_pipe),('clf',clf[i])])



### grid ###



    grid=GridSearchCV(pipe,param_grid=param_grid[i],cv=4,scoring='accuracy')

    g=grid.fit(X_train,y_train)



### prediction ###



## resultats CV

    result=grid.cv_results_



## best parameters 

    bp=grid.best_params_

    print('Best parameters for %s:'%clf_name[i])

    print(bp)



## best estimator 

    be=grid.best_estimator_





### Results ### 



## prediction score

    

    predict=be.predict(X_test)



## scores



    report=metrics.classification_report(y_test,predict)



    conf_mat=metrics.confusion_matrix(y_test,predict)



    print('Reporting for %s:'%clf_name[i])

    print(report)



    print('Confusion matrix for %s:'%clf_name[i])

    print(conf_mat)



### learning curve ###



    from sklearn.model_selection import learning_curve

    from sklearn.model_selection import ShuffleSplit



    cv=ShuffleSplit(n_splits=10,test_size=0.2,train_size=None,random_state=1)





    train_size,train_score,cv_score=learning_curve(be,X_main,

                                                 y_main,

                                                 cv=cv,scoring='accuracy')

    

    train_size_all.append(train_size)

    train_score_all.append(train_score)

    cv_score_all.append(cv_score)



### ROC curve ###



    clf_proba=be.predict_proba(X_test)



    fpr,tpr,thresolds=roc_curve(y_test,clf_proba[:,1])

    

    auc=roc_auc_score(y_test,clf_proba[:,1])

    

    fpr_all.append(fpr)

    tpr_all.append(tpr)

    auc_all.append(auc)



plt.plot(fpr_all[0],tpr_all[0],'g',label='AUC SVM rate=%0.4f'%auc_all[0])

plt.plot(fpr_all[1],tpr_all[1],'b',label='AUC RegLog rate=%0.4f'%auc_all[1])



plt.plot([0,1],[0,1],'k--')

plt.title('ROC Curve')

plt.xlabel('fpr')

plt.ylabel('recall')

plt.legend(loc='lower right')



plt.show()

## learning curve



# train_score all have 3 dimensions: 1. algorithm: 2 values

#                                   2. train size: 5 valeurs

#                                   3. score: n values, n=n_split

train_score_mean=np.mean(train_score_all,axis=2) 

cv_score_mean=np.mean(cv_score_all,axis=2)

    



plt.plot(train_size,train_score_mean[0],marker='+',color='g',label='train score')

plt.plot(train_size,cv_score_mean[0],marker='+',color='b',label='CV score')

plt.title('Learning Curve SVM')

plt.xlabel('Nombre de données du train')

plt.ylabel('Score')

plt.legend(loc='lower right')

plt.grid(True)



plt.show()



plt.plot(train_size,train_score_mean[1],marker='+',color='g',label='train score')

plt.plot(train_size,cv_score_mean[1],marker='+',color='b',label='CV score')

plt.title('Learning Curve RegLog')

plt.xlabel('Nombre de données du train')

plt.ylabel('Score')

plt.legend(loc='lower right')

plt.grid(True)



plt.show()