import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns
from imblearn.over_sampling import SMOTE, ADASYN



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import  OneHotEncoder as ohe

from sklearn.preprocessing import StandardScaler as ss

from sklearn.compose import ColumnTransformer as ct
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier
#import scikitplot as skplt

from sklearn.metrics import confusion_matrix

from sklearn.metrics import average_precision_score

import sklearn.metrics as metrics

from sklearn.metrics import f1_score, recall_score
# Read the CSV file

transactions = pd.read_csv('../input/datasetForFinalAssignment.csv')

print (transactions.shape)

transactions.head()
transactions["class"].value_counts(normalize=True)
sns.countplot(x="class", data=transactions)
transactions["timeBetween"] = (pd.to_datetime(transactions["signup_time"])-pd.to_datetime(transactions["purchase_time"])).dt.seconds



#transactions.apply (lambda row: timeBetween(row), axis=1)

transactions.head(3)
transactions["numberOfTimesDeviceUsed"] = transactions.groupby("device_id")["user_id"].transform("count")

transactions.head(3)
# What is the percentage of the fraudulent transactions occur when a device was used multiple times? 



transactions['class'].value_counts().plot.bar()

transactions['class'].value_counts()



totalFraudulent = len(transactions[transactions['class'] == 1])

print('total number of fraudulent transactions : ', totalFraudulent)



totalNormal = len(transactions[transactions['class'] == 0])

print('total number of Normal transactions : ', totalNormal)



## find the cases where timeBetween is zero (just signed up to make a purchase)

zeroSecondsTransactions = transactions[(transactions['signup_time-purchase_time'] == 0)]

fraudulentAtZeroSeconds = len(zeroSecondsTransactions[zeroSecondsTransactions['class']==1])

nonFraudulentAtZeroSeconds = len(zeroSecondsTransactions[zeroSecondsTransactions['class']==0])



print('total number of fraudulent transactions at zero seconds : ', fraudulentAtZeroSeconds)

print('total number of NONfraudulent transactions at zero seconds : ', nonFraudulentAtZeroSeconds)

print('if the customer makes a purchase at ZERO seconds after signup, these transactions are FRAUDULENT ', round(100*fraudulentAtZeroSeconds/len(zeroSecondsTransactions),2) ,'% of the time' )

ratio = round(100*(fraudulentAtZeroSeconds/totalFraudulent),2)

print('In the given dataset ', ratio,'% of fraudulent transactions happen at ZERO seconds from signup')



deviceUsedMoreThanOnce = transactions[(transactions['N[device_id]'] > 1)]

print(len(deviceUsedMoreThanOnce), ' of the records show that the same device was used more than once')



fraudulentMultipleUse = deviceUsedMoreThanOnce[deviceUsedMoreThanOnce['class'] == 1]

print( ' Of these ', len(deviceUsedMoreThanOnce), ' transactions, ', len(fraudulentMultipleUse), 'of them are fraudulent ---', round(100*len(fraudulentMultipleUse)/len(deviceUsedMoreThanOnce),2), '% of the time')

transactions.info()
dataCopy = transactions.copy()
y = transactions["class"]

#transactions.drop(columns = ["Column 1", "user_id","device_id","class","signup_time","purchase_time","ip_address"], inplace=True)

transactions.drop(columns = ["Column 1", "user_id","device_id","class","signup_time","purchase_time","ip_address"], inplace=True)
transactions.info()
transactions["sex"] = transactions["sex"].map({'M':1,'F':0})
transactions.head(5)
all_columns = transactions.columns
num_columns = transactions[['signup_time-purchase_time','purchase_value','age','sex','N[device_id]']]

stdScalar = ss()

stdScalar.fit(num_columns)

scaled_Transactions = stdScalar.transform(num_columns)

df_scaledTransactions = pd.DataFrame(scaled_Transactions) #, index=scaled_Transactions.index, columns=num_columns)

df_scaledTransactions.columns = ['signup_time-purchase_time','purchase_value','age','sex','N[device_id]']

browser_transactions = pd.get_dummies(transactions.browser, prefix='Browser').iloc[:,1:]

source_transactions = pd.get_dummies(transactions.source, prefix='Source').iloc[:,1:]
X_transAndScaledData = pd.concat([df_scaledTransactions, browser_transactions, source_transactions], axis=1)
X_transAndScaledData.info()
X_transAndScaledData.head()
X_transAndScaledData.shape
X_train, X_test, y_train, y_test,indicies_tr,indicies_test =  train_test_split(X_transAndScaledData,

                                                      y,

                                                      np.arange(X_transAndScaledData.shape[0]),

                                                      train_size = 0.95,

                                                      test_size = 0.05,                                                      

                                                      )
X_train.shape
y_train.head()
dataForCost = dataCopy.iloc[indicies_test]

dataForCost.purchase_value.head()
def normalize(X):

    """

    Make the distribution of the values of each variable similar by subtracting the mean and by dividing by the standard deviation.

    """

    for feature in X.columns:

        X[feature] -= X[feature].mean()

        X[feature] /= X[feature].std()

    return X
sm = SMOTE(random_state=42)

# Normalize the data

#X_train = normalize(X_train)

#X_test = normalize(X_test)

    

X_res, y_res = sm.fit_sample(X_train, y_train)

np.sum(y_res)/len(y_res)
X_res.shape, y_res.shape
def PrintStats(cmat, y_test, pred):

   # separate out the confusion matrix components

   tpos = cmat[0][0]

   fneg = cmat[1][0]

   fpos = cmat[0][1]

   tneg = cmat[1][1]

   print(cmat)

  

   print('True Positive:' + str(tpos))

   print('False Negative:' + str(fneg))

   print('False Positive:' + str(fpos))

   print('True Negative:' + str(tneg))

   

   # calculate F!, Recall scores

   f1Score = round(f1_score(y_test, pred), 2)

   recallScore = round(recall_score(y_test, pred), 2)

   # calculate and display metrics  

   print( 'Accuracy: '+ str(np.round(100*float(tpos+fneg)/float(tpos+fneg + fpos + tneg),2))+'%')

   #print( 'Cohen Kappa: '+ str(np.round(cohen_kappa_score(y_test, pred),3)))

   print("Sensitivity/Recall for Model : {recall_score}".format(recall_score = recallScore))

   print("F1 Score for Model : {f1_score}".format(f1_score = f1Score))
def CalculateCost(actual,prediction,data):

  #if model prediction is true, but it is actually false - this will cost $8 per customer

  costofFalsePositive = data.purchase_value[(prediction==1) & (actual==0)].count() * 8  

  print("Cost for false predicition: ${:.0f}".format(costofFalsePositive))

  #if model prediction is false, but it is actually true - this will cost the purchase value

  costofFalseNegative = data.purchase_value[(prediction==0) & (actual==1)].sum()

  print("Cost lost due to wrong predicution: ${:.0f}".format(costofFalseNegative))

  totalCost = costofFalsePositive + costofFalseNegative

  print("total Cost ${:.0f}".format(totalCost))

  

  #return totalCost
lr = LogisticRegression()

# Fit and predict!

lr.fit(X_train, y_train)

Y_pred = lr.predict(X_test)

lr_cnf_matrix = confusion_matrix(y_test, Y_pred)
# And finally: show the results

print(classification_report(y_test, Y_pred))
PrintStats(lr_cnf_matrix, y_test, Y_pred)
CalculateCost(y_test,Y_pred,dataForCost)
lr = LogisticRegression()

# Fit and predict!

lr.fit(X_res, y_res)

Y_pred = lr.predict(X_test)

lr_cnf_matrix = confusion_matrix(y_test, Y_pred)
# And finally: show the results

print(classification_report(y_test, Y_pred))
PrintStats(lr_cnf_matrix, y_test, Y_pred)

CalculateCost(y_test,Y_pred,dataForCost)
Y_pred.tofile("submission_MinCost.csv",sep=",")
dt = DecisionTreeClassifier()

dt.fit(X_res, y_res)

Y_dt_pred = dt.predict(X_test)

dt_cnf_matrix = confusion_matrix(y_test, Y_dt_pred)
# And finally: show the results

print(classification_report(y_test, Y_dt_pred))
PrintStats(dt_cnf_matrix, y_test, Y_dt_pred)
CalculateCost(y_test, Y_dt_pred, dataForCost)
rf = RandomForestClassifier(n_estimators = 500, n_jobs =4)

rf.fit(X_res, y_res)

Y_rf_pred = rf.predict(X_test)

rf_cnf_matrix = confusion_matrix(y_test, Y_rf_pred)
# And finally: show the results

print(classification_report(y_test, Y_rf_pred))
PrintStats(rf_cnf_matrix, y_test, Y_rf_pred)
CalculateCost(y_test,Y_rf_pred,dataForCost)


#testData =  pd.read_csv('../input/datasetForFinalTest.csv')

#testData.head()
#dataTestCopy = testData.copy()

#testData.drop(columns = ["Column 1", "user_id","device_id","signup_time","purchase_time","ip_address"], inplace=True)

#testData["sex"] = testData["sex"].map({'M':1,'F':0})

#num_columns = testData[['signup_time-purchase_time','purchase_value','age','sex','N[device_id]']]

#stdScalar = ss()

#stdScalar.fit(num_columns)

#scaled_TestTransactions = stdScalar.transform(num_columns)

#df_testscaledTransactions = pd.DataFrame(scaled_TestTransactions) #, index=scaled_Transactions.index, columns=num_columns)

#df_testscaledTransactions.columns = ['signup_time-purchase_time','purchase_value','age','sex','N[device_id]']

#browser_testtransactions = pd.get_dummies(testData.browser, prefix='Browser').iloc[:,1:]

#source_testtransactions = pd.get_dummies(testData.source, prefix='Source').iloc[:,1:]

#X_testtransAndScaledData = pd.concat([df_testscaledTransactions, browser_testtransactions, source_testtransactions], axis=1)
#X_testtransAndScaledData.info()
#X_testtransAndScaledData.head()
#lr = LogisticRegression()

#lr.fit(X_res, y_res)

#Y_pred = lr.predict(X_testtransAndScaledData)
#Y_pred.tofile("submission_FinalTest.csv",sep=",")