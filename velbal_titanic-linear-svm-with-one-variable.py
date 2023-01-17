import pandas            as pd

import numpy             as np

import matplotlib.pyplot as plt



from sklearn                 import svm

from sklearn                 import metrics
# Define parameters

FCSV_TRAIN="../input/train.csv"

FCSV_TESTX="../input/test.csv"

FCSV_TESTY="../input/gender_submission.csv"

Y="Survived"

KERNEL_TYP='linear'

REMOVE  =["PassengerId","Name","Sex","Embarked","Ticket","Cabin","Age"]

# The hyper-parameters for the model which has been constructed by non-linear SVM before

C_HYPERP    =64.0

GAMMA_HYPERP=0.03125
# Load training data from train.csv

data_train=pd.read_csv(FCSV_TRAIN)

data_train=pd.concat([data_train.drop(REMOVE,axis=1),pd.get_dummies(data_train['Sex']),      \

                                         pd.get_dummies(data_train['Embarked'])],axis=1)

data_train=data_train.drop(['female'],axis=1)

data_train=data_train.drop(['C']     ,axis=1)

data_train=data_train.dropna()



# Auto scaling training data with mean=0 and var=1 

mean=data_train.mean(axis=0)

std =data_train.std(axis=0,ddof=1)

data_train['Pclass']=(data_train['Pclass']-mean['Pclass'])/std['Pclass']

data_train['Parch'] =(data_train['Parch'] -mean['Parch']) /std['Parch']

data_train['SibSp'] =(data_train['SibSp'] -mean['SibSp']) /std['SibSp']

data_train['Fare']  =(data_train['Fare']  -mean['Fare'])  /std['Fare']
# Load test data from test.csv and gender_submission.csv

data_testx=pd.read_csv(FCSV_TESTX)

data_testy=pd.read_csv(FCSV_TESTY)

data_test=pd.concat([data_testy,data_testx],axis=1)

data_test=pd.concat([data_test.drop(REMOVE,axis=1),pd.get_dummies(data_test['Sex']),      \

                                                   pd.get_dummies(data_test['Embarked'])],axis=1)

data_test=data_test.drop(['female'],axis=1)

data_test=data_test.drop(['C'],axis=1)

data_test=data_test.dropna()



# Auto scaling test data with mean=0 and var=1 

data_test['Pclass']=(data_test['Pclass']-mean['Pclass'])/std['Pclass']

data_test['Parch'] =(data_test['Parch'] -mean['Parch']) /std['Parch'] 

data_test['SibSp'] =(data_test['SibSp'] -mean['SibSp']) /std['SibSp'] 

data_test['Fare']  =(data_test['Fare']  -mean['Fare'])  /std['Fare'] 
x_names_pd=pd.DataFrame(data_train.drop([Y],axis=1).columns)

x_names_np=np.array(x_names_pd)

y_train_np=np.array(data_train[Y])

y_test_np =np.array(data_test[Y])

acc_all_train_np=np.array([])

acc_all_test_np =np.array([])



for i,x_name in enumerate(x_names_np):

   x_train_np =np.array([])

   x_test_np  =np.array([])

   yp_train_np=np.array([])    

   yp_test_np =np.array([])    



   x_train_np=np.array(data_train[x_name]).reshape(-1,1)

   x_test_np =np.array(data_test[x_name]).reshape(-1,1)  

   model=svm.SVC(kernel=KERNEL_TYP,C=C_HYPERP)

   model.fit(x_train_np,y_train_np)

   yp_train_np=model.predict(x_train_np)

   yp_test_np =model.predict(x_test_np)

   acc_train=metrics.accuracy_score(y_train_np,yp_train_np)

   acc_test =metrics.accuracy_score(y_test_np ,yp_test_np) 

   acc_all_train_np=np.append(acc_all_train_np,acc_train) 

   acc_all_test_np =np.append(acc_all_test_np ,acc_test) 
acc_all_train_pd=pd.DataFrame(acc_all_train_np)

acc_all_test_pd =pd.DataFrame(acc_all_test_np)



results_pd=pd.DataFrame([])

results_pd=pd.concat([acc_all_train_pd,acc_all_test_pd],axis=1)

results_pd.columns=['Training data','Test data']

results_pd.index  =x_names_pd



results_pd.plot(kind='bar',alpha=0.6,figsize=(12,3),grid=True,title='Explanatory variable (x-axis) vs Accuracy (y-axis) of SVM with linear kernel')