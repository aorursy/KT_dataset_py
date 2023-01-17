import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("darkgrid")

data  = pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")
data  = pd.DataFrame(data)
print("DataFrame shape :",data.shape)
# missing_values in dataframe
data.isnull().sum()/len(data)*100
data.head()
fraud_amount = data[data.Class == 1]["Amount"]
fraud_amount = fraud_amount.astype(int)
fraud_amount.hist(color = "r",alpha = 0.6)
plt.xlabel("Fraud Amount")
plt.ylabel("Frequency")
print("Highest Fruad amount was :",max(fraud_amount))
print("Least   Fruad amount was :",min(fraud_amount))
# i have no idea  how they  , considered amount 0 as fruad transcation .!! Because there is no transcation at all

# most of the fraud amount was below 1000
fraud_amount.hist(color = "b",alpha = 0.3,bins = [1,1000,2000])
plt.xlabel("Fraud Amount")
plt.ylabel("Frequency")
# our Data seems imbalanced 
sns.countplot(data["Class"],color = "orange")
print(data["Class"].value_counts())
print("="*60)
print("Percentage of class values :")
print(data["Class"].value_counts()/len(data)*100)
data["Amount"].hist(color = "y",bins = [0,500,1500,2000])

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
data["scaled_amount"] = sc.fit_transform(np.array(data["Amount"]).reshape(-1,1))
data.drop(["Amount","Time"],axis = 1,inplace = True)

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
def train_set(ytrain,train_pred):
    print("confusion matrix for train set : ")
    cm = confusion_matrix(ytrain,train_pred)
    print(cm)
    print("--"*40)
    print("False positive rate :",(cm[1][0]/(cm[1][0]+cm[1][1]))*100)  # FPR = FP/FP +TN
    print("\n")
    print(cm[1][0] ,"out of",(cm[1][0]+cm[1][1]),"fraud transaction instances were classified as not a fraudulent transactions \n")
    print("--"*40)
    print("False Negative rate :",(cm[0][1]/(cm[0][1]+cm[0][0]))*100) # FNR = FN + (FN + TP )
    print("\n")
    print(cm[0][1],"out of ",(cm[0][1]+cm[0][0]),"non fraudulent transaction instances were classified as  a fraudulent transactions")
    print("--"*40)
    print(classification_report(ytrain,train_pred))
def test_set(ytest,test_pred):
    print("confusion matrix for test set : ")
    cm = confusion_matrix(ytest,test_pred)
    print(cm)
    print("--"*40)
    print("False positive rate :",(cm[1][0]/(cm[1][0]+cm[1][1]))*100)  # FPR = FP/FP +TN
    print("\n")
    print(cm[1][0] ,"out of",(cm[1][0]+cm[1][1]),"fraud transaction instances were classified as not a fraudulent transactions \n")
    print("--"*40)
    print("False Negative rate :",(cm[0][1]/(cm[0][1]+cm[0][0]))*100) # FNR = FN + (FN + TP )
    print("\n")
    print(cm[0][1],"out of ",(cm[0][1]+cm[0][0]),"non fraudulent transaction instances were classified as  a fraudulent transactions")
    print("--"*40)
    print(classification_report(ytest,test_pred))
def under_sample(data):
    under_sample_zero = data[data.Class == 0].iloc[:,:]
    under_sample_zero = under_sample_zero.sample(data.Class.value_counts()[1])
    under_sample_one  = data[data.Class == 1].iloc[:,:]
    under_sampled_data = pd.concat([under_sample_zero,under_sample_one],axis = 0)
    us_x = under_sampled_data.loc[:,under_sampled_data.columns != "Class"]
    us_y = under_sampled_data.loc[:,"Class"]
    return us_x,us_y,under_sample_zero,under_sample_one,under_sampled_data

us_x,us_y,under_sample_zero,under_sample_one,under_sampled_data = under_sample(data)
# Now we have equal number of 1's and 0's ,i.e we have balanced data !

print("under_sampled_data size      : ",under_sampled_data.shape)
print("sample size where class is 0 :",under_sample_zero.shape)
print("sample size where class is 1 :",under_sample_one.shape)
print("Features shape :",x.shape)
print("Dependent Variable shape ",y.shape)
sns.countplot(under_sampled_data.Class)
plt.show()
x = data.loc[:,data.columns != "Class"]
y = data.loc[:,"Class"]
print("Features shape :",x.shape)
print("Dependent Variable shape ",y.shape)

lr_model = LogisticRegression()
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size = 0.4,random_state = 1)
lr_model.fit(xtrain,ytrain)
#train_set
train_pred = lr_model.predict(xtrain)
#test_set
test_pred  = lr_model.predict(xtest)
print("Accuracy for Training set : ",accuracy_score(ytrain,train_pred))
print("Accuracy for Testing set  : ",accuracy_score(ytest,test_pred))

train_set(ytrain,train_pred)
test_set(ytest,test_pred)
lr_model = LogisticRegression()
us_x,us_y,under_sample_zero,under_sample_one,under_sampled_data = under_sample(data)
xtrain,xtest,ytrain,ytest = train_test_split(us_x,us_y,test_size = 0.4,random_state = 1)
lr_model.fit(xtrain,ytrain)
#train_set
train_pred = lr_model.predict(xtrain)
#test_set
test_pred  = lr_model.predict(xtest)
print("Accuracy for Training set : ",accuracy_score(ytrain,train_pred))
print("Accuracy for Testing set  : ",accuracy_score(ytest,test_pred))
roc_auc_logistic = roc_auc_score(ytest,test_pred)

train_set(ytrain,train_pred)
# on applying undersampling technique we got !
# Decrease in False positive rate ,
# Increase in precision,recall,f1-score 
test_set(ytest,test_pred)
# on applying undersampling technique we got !
# Decrease in False positive rate ,
# Increase in precision,recall,f1-score 

rf_model = RandomForestClassifier(n_estimators=20,max_depth = 10,min_samples_split = 20)
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size = 0.4,random_state = 1)
rf_model.fit(xtrain,ytrain)
#train_set
train_pred = rf_model.predict(xtrain)
#test_set
test_pred  = rf_model.predict(xtest)

print("Accuracy for Training set : ",accuracy_score(ytrain,train_pred))
print("Accuracy for Testing set  : ",accuracy_score(ytest,test_pred)) 
train_set(ytrain,train_pred)
# 57 out of 306 fraud transaction instances were classified as non a fraudulent transactions 
test_set(ytest,test_pred)
# 51 out of 186 fraud transaction instances were classified as not a fraudulent transactions
# 25% of fradulent transaction were classified as non fraudulent transactions
us_x,us_y,under_sample_zero,under_sample_one,under_sampled_data = under_sample(data)
rf_model = RandomForestClassifier(n_estimators=20,max_depth = 10,min_samples_split = 20)
xtrain,xtest,ytrain,ytest = train_test_split(us_x,us_y,test_size = 0.4,random_state = 1)
rf_model.fit(xtrain,ytrain)
#train_set
train_pred = rf_model.predict(xtrain)
#test_set
test_pred  = rf_model.predict(xtest)
roc_auc_random = roc_auc_score(ytest,test_pred)
train_set(ytrain,train_pred)
# on applying undersampling technique we got !
# Decrease in False positive rate ,
# Increase in precision,recall,f1-score 
test_set(ytest,test_pred)
# on applying undersampling technique we got !
# Decrease in False positive rate ,
# Increase in precision,recall,f1-score 
# for test_set's only (validation set)
print("ROC_AUC_SCORE FOR LOGISTIC REG IS  : ",roc_auc_logistic)
print("ROC_AUC_SCORE FOR Random Forest IS : ",roc_auc_random)
# both algo's are working equally ,
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state = 1)
over_sam_x,over_sam_y = sm.fit_sample(x,y)
print("Feature size   after oversampling :",over_sam_x.shape)
print("Dependent size after oversampling :",over_sam_y.shape)
sns.countplot(over_sam_y)


xtrain,xtest,ytrain,ytest = train_test_split(over_sam_x,over_sam_y,test_size = 0.4,random_state = 1)
lr_model.fit(xtrain,ytrain)
#train_set
train_pred = lr_model.predict(xtrain)
#test_set
test_pred  = lr_model.predict(xtest)
print("Accuracy for Training set : ",accuracy_score(ytrain,train_pred))
print("Accuracy for Testing set  : ",accuracy_score(ytest,test_pred))
over_sam_roc_auc_logistic = roc_auc_score(ytest,test_pred)
train_set(ytrain,train_pred)
# on applying undersampling technique we got !
# Decrease in False positive rate ,
# Increase in precision,recall,f1-score 
test_set(ytest,test_pred)
# on applying undersampling technique we got !
# Decrease in False positive rate ,
# Increase in precision,recall,f1-score
rf_model = RandomForestClassifier(n_estimators=20,max_depth = 10,min_samples_split = 20)
xtrain,xtest,ytrain,ytest = train_test_split(over_sam_x,over_sam_y,test_size = 0.4,random_state = 1)
rf_model.fit(xtrain,ytrain)
#train_set
train_pred = rf_model.predict(xtrain)
#test_set
test_pred  = rf_model.predict(xtest)
print("Accuracy for Training set : ",accuracy_score(ytrain,train_pred))
print("Accuracy for Testing set  : ",accuracy_score(ytest,test_pred))
over_sam_roc_auc_random_forest = roc_auc_score(ytest,test_pred)
train_set(ytrain,train_pred)
# on applying undersampling technique we got !
# Decrease in False positive rate ,
# Increase in precision,recall,f1-score 
test_set(ytest,test_pred)
# on applying undersampling technique we got !
# Decrease in False positive rate ,
# Increase in precision,recall,f1-score
# for test_set's only 
print("ROC_AUC_SCORE FOR LOGISTIC REG IS  : ",over_sam_roc_auc_logistic)
print("ROC_AUC_SCORE FOR Random Forest IS : ",over_sam_roc_auc_random_forest)
#  Random Forest performs well compared to Logistic Regression.
