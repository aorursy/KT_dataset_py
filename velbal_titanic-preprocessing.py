import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt
import seaborn           as sns
FCSV_TRAIN="../input/train.csv"
FCSV_TESTX="../input/test.csv"
FCSV_TESTY="../input/gender_submission.csv"
Y="Survived"
REMOVE=["PassengerId","Name","Sex","Embarked","Ticket","Cabin","Age"]
# Load training data from train.csv
data_train=pd.read_csv(FCSV_TRAIN)
data_train=pd.concat([data_train.drop(REMOVE,axis=1),pd.get_dummies(data_train['Sex']),      \
                                                     pd.get_dummies(data_train['Embarked'])],axis=1)
data_train=data_train.drop(['female'],axis=1)
data_train=data_train.drop(['C']     ,axis=1)
data_train=data_train.dropna()
data_train_dead =data_train[data_train['Survived']==0]
data_train_alive=data_train[data_train['Survived']==1]
# Show histgrams for seven types of descriptors at training data 
sns.set()
sns.set_style('whitegrid')
sns.set_palette('Set1')
column_names_train=data_train.columns
fig,axes=plt.subplots(nrows=2,ncols=4,figsize=(15,8))

n=1
for i in range(2): 
   for j in range(4): 
      if i==1 and j==3: 
         axes[i,j].axis('off')
      else:          
         name=column_names_train[n]
         axes[i,j].set_xlabel(name)
         axes[i,j].hist(data_train_dead[name] ,alpha=0.6,bins=30)
         axes[i,j].hist(data_train_alive[name],alpha=0.6,bins=30)
         n+=1   
plt.tight_layout()
plt.show()    
# Load test data from test.csv and gender_submission.csv
data_testx=pd.read_csv(FCSV_TESTX)
data_testy=pd.read_csv(FCSV_TESTY)
data_test=pd.concat([data_testy,data_testx],axis=1)
data_test=pd.concat([data_test.drop(REMOVE,axis=1),pd.get_dummies(data_test['Sex']),      \
                                                   pd.get_dummies(data_test['Embarked'])],axis=1)
data_test=data_test.drop(['female'],axis=1)
data_test=data_test.drop(['C'],axis=1)
data_test=data_test.dropna()
data_test_dead =data_test[data_test['Survived']==0]
data_test_alive=data_test[data_test['Survived']==1]
# Show histgrams for seven types of descriptors at test data 
sns.set()
sns.set_style('whitegrid')
sns.set_palette('Set1')
column_names_test=data_test.columns
fig,axes=plt.subplots(nrows=2,ncols=4,figsize=(15,8))

n=1
for i in range(2): 
   for j in range(4): 
      if i==1 and j==3: 
         axes[i,j].axis('off')
      else:          
         name=column_names_test[n]
         axes[i,j].set_xlabel(name)
         axes[i,j].hist(data_test_dead[name] ,alpha=0.6,bins=30)
         axes[i,j].hist(data_test_alive[name],alpha=0.6,bins=30)
         n+=1   
plt.tight_layout()
plt.show()    