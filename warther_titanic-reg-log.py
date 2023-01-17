### IMPORT



import csv as csv

import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

%matplotlib inline



main_df=pd.read_csv("../input/train.csv",dtype={"Age": np.float64},)



test_df=pd.read_csv("../input/test.csv",dtype={"Age": np.float64},)



passengerID_test=test_df.PassengerId
print(main_df.head())

print(test_df.head())

print('MAIN INFO')

print(main_df.info())

print('TEST INFO')

print(test_df.info())
main_df=main_df.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)



test_df=test_df.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)
## Get_dummies



dum1_main=pd.get_dummies(main_df[['Sex']])

dum2_main=pd.get_dummies(main_df[['Embarked']])



dum1_test=pd.get_dummies(test_df[['Sex']])

dum2_test=pd.get_dummies(test_df[['Embarked']])



## Concaténation



main_df=pd.concat([dum1_main,dum2_main,main_df],axis=1)



test_df=pd.concat([dum1_test,dum2_test,test_df],axis=1)



## Drop



main_df=main_df.drop(['Sex','Embarked','Sex_female'],axis=1)

test_df=test_df.drop(['Sex','Embarked','Sex_female'],axis=1)

## Transform NaN





for i in range(len(main_df)):

    if pd.isnull(main_df.loc[i,'Age'])==True:

        main_df.loc[i,'Age']=-1



for i in range(len(test_df)):

    if pd.isnull(test_df.loc[i,'Age'])==True:

        test_df.loc[i,'Age']=np.mean(test_df['Age'])

      

for i in range(len(test_df)):

    if pd.isnull(test_df.loc[i,'Fare'])==True:

        test_df.loc[i,'Fare']=np.mean(test_df['Fare'])

 

   

## Transform Null



for i in range(len(main_df)):

    if main_df.loc[i,'Fare']==0:

        main_df.loc[i,'Fare']=-1

   

for i in range(len(test_df)):

    if test_df.loc[i,'Fare']==0:

        test_df.loc[i,'Fare']=np.mean(test_df['Fare'])
print('MAIN INFO')

print(main_df.info())

print('TEST INFO')

print(test_df.info())
#### Visu ####



corr=main_df.corr()

plt.matshow(corr)



y_label=main_df.columns.values

y_pos=np.arange(len(y_label))

x_label=y_label.copy()

for i in range(len(x_label)):

    x_label[i]=y_label[i][:4]

x_pos=np.arange(len(x_label))

plt.xticks(x_pos,x_label, rotation=90)

plt.yticks(y_pos,y_label)

plt.colorbar()



plt.show()
main_df.corr()["Survived"]



## Age





main_df.loc[(main_df['Age']<=4) & (main_df['Age']>0),'Age']=1

main_df.loc[(main_df['Age']<=14) & (main_df['Age']>4),'Age']=2

main_df.loc[(main_df['Age']<=25) & (main_df['Age']>14),'Age']=3

main_df.loc[(main_df['Age']<=45) & (main_df['Age']>25),'Age']=4

main_df.loc[(main_df['Age']<=60) & (main_df['Age']>45),'Age']=5

main_df.loc[(main_df['Age']>60),'Age']=6





test_df.loc[(test_df['Age']<=4) & (test_df['Age']>0),'Age']=1

test_df.loc[(test_df['Age']<=14) & (test_df['Age']>4),'Age']=2

test_df.loc[(test_df['Age']<=25) & (test_df['Age']>14),'Age']=3

test_df.loc[(test_df['Age']<=45) & (test_df['Age']>25),'Age']=4

test_df.loc[(test_df['Age']<=60) & (test_df['Age']>45),'Age']=5

test_df.loc[test_df['Age']>60,'Age']=6







## Family Size



main_df['FamilySize']=main_df['Parch']+main_df['SibSp']+1



test_df['FamilySize']=test_df['Parch']+test_df['SibSp']+1





## Alone



main_df['IsAlone']=0

main_df.loc[main_df['FamilySize']==1,'IsAlone']=1



test_df['IsAlone']=0

test_df.loc[test_df['FamilySize']==1,'IsAlone']=1



## Men alone class 3



main_df['MenAloneC3']=0

main_df.loc[(main_df['Sex_male']==1) & (main_df['Pclass']==3) & (main_df['IsAlone']==1),'MenAloneC3']=1



test_df['MenAloneC3']=0

test_df.loc[(test_df['Sex_male']==1) & (test_df['Pclass']==3) & (test_df['IsAlone']==1),'MenAloneC3']=1



## Women alone class 3



main_df['WomenAloneC3']=0

main_df.loc[(main_df['Sex_male']==0) & (main_df['Pclass']==3) & (main_df['IsAlone']==1),'WomenAloneC3']=1



test_df['WomenAloneC3']=0

test_df.loc[(test_df['Sex_male']==0) & (test_df['Pclass']==3) & (test_df['IsAlone']==1),'WomenAloneC3']=1



## Women class 1-2



main_df['WomenC12']=0

main_df.loc[(main_df['Sex_male']==0) & (main_df['Pclass']!=3),'WomenC12']=1



test_df['WomenC12']=0

test_df.loc[(test_df['Sex_male']==0) & (test_df['Pclass']!=3),'WomenC12']=1



## Child class 1-2



main_df['ChildC12']=0

main_df.loc[(main_df['Age']==1) & (main_df['Pclass']!=3),'ChildC12']=1



test_df['ChildC12']=0

test_df.loc[(test_df['Age']==1) & (test_df['Pclass']!=3),'ChildC12']=1



print(test_df.head())
corr=main_df.corr()

plt.matshow(corr)



y_label=main_df.columns.values

y_pos=np.arange(len(y_label))

x_label=y_label.copy()

for i in range(len(x_label)):

    x_label[i]=y_label[i][:4]

x_pos=np.arange(len(x_label))

plt.xticks(x_pos,x_label, rotation=90)

plt.yticks(y_pos,y_label)

plt.colorbar()



plt.show()
main_df.corr()["Survived"]
class_age=[-1,1,2,3,4,5,6]

proportion_age=[]



Tab1=pd.crosstab(main_df['Survived'],main_df['Age'])

print(Tab1)



class_age=[-1,1,2,3,4,5,6]

proportion_age=[]



for i in class_age:

    if i==-1:    

        prop=str(round(Tab1.loc[1,i]/Tab1.loc[0,i],2))

    else:

        prop=str(round(Tab1.loc[1,i]/Tab1.loc[0,i],2))

    proportion_age.append(prop)

    

print(proportion_age)    

 
X_main=main_df.drop(['Survived'],axis=1)

y_main=main_df['Survived']



from sklearn.model_selection import train_test_split



X_train,X_test,y_train,y_test=train_test_split(X_main,y_main,

                                               test_size=0.25,random_state=1)



print(X_train.info())

print(X_test.info())



from sklearn.preprocessing import Imputer

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import PolynomialFeatures





from sklearn.pipeline import Pipeline



from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.svm import SVC





poly=PolynomialFeatures(1)

X_train=poly.fit_transform(X_train)

X_test=poly.fit_transform(X_test)

test_df=poly.fit_transform(test_df)



scale=StandardScaler()



imput=Imputer(missing_values=-1 ,strategy='mean')
### Algorithm ###



clf=LogisticRegression()



### Pipeline ###



pipe=Pipeline([('imput',imput),('poly',poly),('scale',scale),('clf',clf)])



### Grid search ###



from sklearn.model_selection import GridSearchCV



param_grid=dict(clf__C=[0.001,0.01,0.1,1,10,50,100])



grid=GridSearchCV(pipe,param_grid=param_grid,cv=3,scoring='accuracy')



grid.fit(X_train,y_train)



bp=grid.best_params_

be=grid.best_estimator_



print("best parameters are:",bp)



print("best score is:",grid.best_score_)
### ROC ###



from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score



clf_proba=be.predict_proba(X_test)



fpr,tpr,thresolds=roc_curve(y_test,clf_proba[:,1])

    

auc=roc_auc_score(y_test,clf_proba[:,1])

    

## plot



plt.figure(2)



plt.plot(fpr,tpr,'g',label='AUC SVM rate=%0.4f'%auc)



plt.plot([0,1],[0,1],'k--')

plt.title('ROC Curve')

plt.xlabel('fpr')

plt.ylabel('recall')

plt.legend(loc='lower right')



plt.show()

#### Plot  ####     



from sklearn import metrics



predict=be.predict(X_test)



conf_mat=metrics.confusion_matrix(y_test,predict)

print(conf_mat)
