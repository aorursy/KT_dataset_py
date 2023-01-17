

#First we import all the libraries

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import pandas as pd

from scipy import stats

from sklearn.utils import resample
target = 'failure'   #defining a global variable


df_train = pd.read_csv("/kaggle/input/hard-drive-data-and-stats/data_Q3_2019/data_Q3_2019/2019-07-09.csv")  #Training Dataset



df_test = pd.read_csv("/kaggle/input/hard-drive-data-and-stats/data_Q3_2019/data_Q3_2019/2019-07-10.csv")    #Test Dataset

df_train.head()
df_train.describe()
df_test.head()
df_test.describe()
print(df_train.shape)

print('*'*50)

print(df_test.shape)
sns.countplot(df_train['failure'])    #Checking the distribution of the target variable


valid = df_train[df_train['failure'] == 0]    #data of HDDs which do not indicate failure

failed = df_train[df_train['failure'] == 1]   #data of HDDs likely to fail



print("valid hdds:",len(valid))      #storing the total number of valid HDDs

print("failing hdds:",len(failed))    #storing the total number of HDDs likely to fail
#Since the number of HDDs indicating failure are too low, we proceed to upsample the minority class viz.'failure'



# We perform this step to prevent our final model from being biased



   #Resampling of the failure class to match the length of valid HDDs



failed_up = resample(failed,replace=True,n_samples=len(valid),random_state=27)  
#Finally we concatenate our newly resampled classes with our training data



df_train = pd.concat([valid,failed_up])

df_train.failure.value_counts()       #Levelling the count of both classes
sns.countplot(df_train['failure'])
df_train
df_train.shape  # You can notice the dimensions have doubled for our training dataset
df_train.isnull().sum()
# features which highly correlate to HDD failure as per BackBlaze 



# https://www.backblaze.com/blog/what-smart-stats-indicate-hard-drive-failures/



# SMART 5 		Reallocated Sectors Count

# SMART 187 		Reported Uncorrectable Errors

# SMART 188 		Command Timeout

# SMART 197 		Current Pending Sector Count

# SMART 198 		Uncorrectable Sector Count
features = ['date',

 'serial_number',

 'model',

 'capacity_bytes',

 'failure',

'smart_5_raw','smart_187_raw','smart_188_raw','smart_197_raw','smart_198_raw']
misc_feat = [fname for fname in df_train if fname not in features]  #misc features to be dropped 

misc_feat
df_train.drop(misc_feat,inplace=True,axis=1)  #Dropping the misc features
df_train
# Since our model cannot proccess string values, we remove the columns which contain string values/object values 

# to avoid errors



obj = df_train.dtypes[df_train.dtypes == object ].index  

obj
df_train = df_train.drop(obj,axis=1)
df_train.isnull().sum()  #Total number of missing values 
#After going through the dataset,i found that the drives which were missing values did not correlate to its failure



#  i.e all drives indicating failure did not contain missing values



# Hence i replaced them with the most commonly occuring values for the respective SMART attributes



df_train['smart_187_raw'] = df_train['smart_187_raw'].fillna(0)  
df_train['smart_5_raw'] = df_train['smart_5_raw'].fillna(0)
df_train['smart_188_raw'] = df_train['smart_188_raw'].fillna(0)
df_train['smart_197_raw'] = df_train['smart_197_raw'].fillna(0)
df_train['smart_198_raw'] = df_train['smart_198_raw'].fillna(0)
df_train.isnull().sum()
df_train = df_train.drop('capacity_bytes',axis=1)
df_train
X_train = df_train.drop('failure',axis=1)

Y_train = df_train['failure']
valid_test = df_test[df_test['failure'] == 0]

failed_test = df_test[df_test['failure'] == 1]



print("valid hdds:",len(valid_test))

print("failing hdds:",len(failed_test))
failed_up_test = resample(failed,replace=True,n_samples=len(valid),random_state=27) #Same steps as in Training data
df_test = pd.concat([valid_test,failed_up_test])

df_test.failure.value_counts()
df_test.head()
df_test.shape
df_test.drop(misc_feat,inplace=True,axis=1) #Since we have the imp features, we move ahead to drop the misc ones
df_test
df_test['smart_187_raw'] = df_test['smart_187_raw'].fillna(0)
df_test['smart_5_raw'] = df_test['smart_5_raw'].fillna(0)
df_test['smart_188_raw'] = df_test['smart_188_raw'].fillna(0)
df_test['smart_197_raw'] =df_test['smart_197_raw'].fillna(0)
df_test['smart_198_raw'] = df_test['smart_198_raw'].fillna(0)
df_test.isnull().sum()
df_test = df_test.drop(obj,axis=1)
df_test = df_test.drop('capacity_bytes',axis=1)
df_test
X_test = df_test.drop('failure',axis=1)

Y_test = df_test['failure']
df_test.shape
 #Building the Random Forest Classifier (RANDOM FOREST) 

from sklearn.ensemble import RandomForestClassifier 



# random forest model creation 

rfc = RandomForestClassifier() 

rfc.fit(X_train, Y_train) 



# predictions(Notice the caps'P' of yPred to differentiate between model 1 and 2) 

yPred = rfc.predict(X_test) 
#Results of our predictions



from sklearn.metrics import classification_report, accuracy_score  

from sklearn.metrics import precision_score, recall_score 

from sklearn.metrics import f1_score, matthews_corrcoef 

from sklearn.metrics import confusion_matrix 

  

n_outliers = len(failed) 

n_errors = (yPred != Y_test).sum()        #Notice the Y_test from iii) of Test Data

print("Model used is: Random Forest classifier") 

  

acc = accuracy_score(Y_test, yPred) 

print("The accuracy is {}".format(acc)) 

  

prec = precision_score(Y_test, yPred) 

print("The precision is {}".format(prec)) 

  

rec = recall_score(Y_test, yPred) 

print("The recall is {}".format(rec)) 

  

f1 = f1_score(Y_test, yPred) 

print("The F1-Score is {}".format(f1)) 

  

MCC = matthews_corrcoef(Y_test, yPred) 

print("The Matthews correlation coefficient is {}".format(MCC)) 

 # confusion matrix 



LABELS = ['Healthy', 'Failed'] 

conf_matrix = confusion_matrix(Y_test, yPred) 

plt.figure(figsize =(12, 12)) 

sns.heatmap(conf_matrix, xticklabels = LABELS,  

            yticklabels = LABELS, annot = True, fmt ="d"); 

plt.title("Confusion matrix") 

plt.ylabel('True class') 

plt.xlabel('Predicted class') 

plt.show() 
xData = X_train.values

yData = Y_train.values
from sklearn.model_selection import train_test_split 



# Splitting of data into training and testing sets 

xTrain, xTest, yTrain, yTest = train_test_split( 

        xData, yData, test_size = 0.2, random_state = 42) 


#Random Forest Classifier 

from sklearn.ensemble import RandomForestClassifier 



# RF model creation 

rfc = RandomForestClassifier() 

rfc.fit(xTrain, yTrain) 



# predictions (notice the small 'p' to differentiate from model 1) 

ypred = rfc.predict(xTest) 

from sklearn.metrics import classification_report, accuracy_score  

from sklearn.metrics import precision_score, recall_score 

from sklearn.metrics import f1_score, matthews_corrcoef 

from sklearn.metrics import confusion_matrix 

  

n_outliers = len(failed) 

n_errors = (ypred != yTest).sum()                             #yTest from the Train_Test_Split function

print("Model used is : Random Forest classifier") 

  

acc = accuracy_score(yTest, ypred) 

print("The accuracy is {}".format(acc)) 

  

prec = precision_score(yTest, ypred) 

print("The precision is {}".format(prec)) 

  

rec = recall_score(yTest, ypred) 

print("The recall is {}".format(rec)) 

  

f1 = f1_score(yTest, ypred) 

print("The F1-Score is {}".format(f1)) 

  

MCC = matthews_corrcoef(yTest, ypred) 

print("The Matthews correlation coefficient is{}".format(MCC)) 

# confusion matrix 



LABELS = ['Normal', 'Failed'] 

conf_matrix = confusion_matrix(yTest, ypred) 

plt.figure(figsize =(12, 12)) 

sns.heatmap(conf_matrix, xticklabels = LABELS,  

            yticklabels = LABELS, annot = True, fmt ="d"); 

plt.title("Confusion matrix") 

plt.ylabel('True class') 

plt.xlabel('Predicted class') 

plt.show() 