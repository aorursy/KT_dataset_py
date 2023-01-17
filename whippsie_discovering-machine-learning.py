import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



from sklearn.datasets import make_blobs

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier 

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.svm import SVC # SVM

from imblearn.over_sampling import SMOTE

from sklearn.metrics import f1_score,confusion_matrix,accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.model_selection import GridSearchCV # for tunnig hyper parameter it will use all combination of given parameters

from sklearn.model_selection import RandomizedSearchCV # same for tunning hyper parameter but will use random combinations of parameters

from sklearn.metrics import confusion_matrix,recall_score,precision_recall_curve,auc,roc_curve,roc_auc_score,classification_report



%matplotlib inline
data = pd.read_csv("../input/creditcard.csv",header = 0)

data.head()
## Pre-processing the data

## Normalizing the amount column



from sklearn.preprocessing import StandardScaler

data['normAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))

data = data.drop(['Time','Amount'],axis=1)

data.head()
## Since the data is largely imbalanced we need to resample the data such that the proportion/ratio between fraudulent and normal transactions are relativeley similar.



x = data.loc[:, data.columns != 'Class']

y = data.loc[:, data.columns == 'Class']
#UNDERSAMPLING

# Number of fraudelent transaction in the existing data

numberOffraudulentTransaction = len(data[data.Class == 1])

fraudIndices = np.array(data[data.Class == 1].index)



# Picking the indices of the normal classes

normalIndices = data[data.Class == 0].index



# Out of the indices we picked, randomly select "x" number (number_records_fraud)

random_normal_indices = np.random.choice(normalIndices, numberOffraudulentTransaction, replace = False)

random_normal_indices = np.array(random_normal_indices)

#UNDERSAMPLING

# Appending the 2 indices

under_sample_indices = np.concatenate([fraudIndices,random_normal_indices])



# Under sample dataset

under_sample_data = data.iloc[under_sample_indices,:]



x_undersample = under_sample_data.loc[:, under_sample_data.columns != 'Class']

y_undersample = under_sample_data.loc[:, under_sample_data.columns == 'Class']



# Showing ratio

print("Percentage of normal transactions: ", len(under_sample_data[under_sample_data.Class == 0])/len(under_sample_data))

print("Percentage of fraud transactions: ", len(under_sample_data[under_sample_data.Class == 1])/len(under_sample_data))

print("Total number of transactions in resampled data: ", len(under_sample_data))
#OVERSAMPLING

## Splitting the data into Training,Validation and Test Set##

## Test Set needs to be unused till the mere end##

X_train, X_test, Y_train, Y_test = train_test_split(data,y, test_size=0.25, random_state=42)

X_train, X_val, Y_train, Y_val = train_test_split(X_train,Y_train, test_size=0.25, random_state=42)

# #Figuring out the ratio of normal transction and fraudelent transaction from training data# #





normal_tdata = X_train[X_train["Class"]==0]

print("train data: length of normal data",len(normal_tdata))

fraud_tdata = X_train[X_train["Class"]==1]

print("train data: length of fraud data",len(fraud_tdata))

## dataset for validation set ##

normal_vdata = X_val[X_val["Class"]==0]

print("For Validation Set :length of normal data",len(normal_vdata))

fraud_vdata = X_val[X_val["Class"]==1]

print("For Validation Set :length of fraud data",len(fraud_vdata))



#SMOTE

#Since the data is highly imbalanced we use the sklearn package to balance out the data by introducing more fraudulent data ##

#basically oversampling of data 

sm = SMOTE(random_state=12, ratio = 'auto', k_neighbors=5)

#Possible ratios : minority, majority, not minority, all, auto

x_train_res, y_train_res = sm.fit_sample(X_train, Y_train.values.ravel())



a = x_train_res[:,28]

b= np.count_nonzero(a == 1)

c= np.count_nonzero(a == 0)

print("length of oversampled data is ",len(x_train_res))

print("Number of normal transcation in oversampled data",b)

print("No.of fraud transcation",c)

print("Proportion of Normal data in oversampled data is ",c/len(x_train_res))

print("Proportion of fraud data in oversampled data is ",b/len(x_train_res))
print ("UNDERSAMPLING")

df = under_sample_data

#train, validate, test = np.split(df.sample(frac=1), [int(.5*len(df)), int(.75*len(df))])

x, x_test, y, y_test = train_test_split(x_undersample,y_undersample,test_size=0.25,train_size=0.75)

x_train, x_cv, y_train, y_cv = train_test_split(x,y,test_size = 0.33,train_size =0.66)



scaler = StandardScaler()



#Get mean+average and standardize to Z

x_train = scaler.fit_transform (x_train)



#Apply same transformation to hidden data

x_cv = scaler.transform (x_cv)

x_test = scaler.transform (x_test)



# cross-validate needs to be here (after the splitting for proper X-V)
#UNDERSAMPLING

# My logic is regressing, guys!



logi = LogisticRegression(class_weight='balanced')

mdl = logi.fit(x_train, y_train.values.ravel())

predictions = logi.predict(x_test)

print("ACCURACY : ",accuracy_score(y_test, predictions))

print ("CMATRIX : ")

print(confusion_matrix(y_test, predictions))

print (classification_report(y_test, predictions))

print("LOGICREGRESS")



'''

# Neural network, captain!

lr = LogisticRegression(C = 1, penalty = 'l1')

lr.fit(x_train, y_train.values.ravel())

y_pred_test_nn = lr.predict(x_test)

print("NEURALNETWORK")

print(accuracy_score(y_test, predictions))

print (confusion_matrix(y_test, predictions))

print (classification_report(y_test, predictions))

'''



# I will not put the receiver operating characteristic, no sir!

'''

# Support vector machine, boss!

print("SVM")



#Other models doing 75%

svc = SVC(C=1, kernel='linear')

svc.fit (x_train,y_train.values.ravel())

ypredsvc = svc.predict (x_test)

print(confusion_matrix(y_test, ypredsvc))

print (classification_report(y_test, predictions))

print(f1_score(y_test, ypredsvc))

'''



# Random Forest stories, mate!



classif = RandomForestClassifier(n_estimators=100, n_jobs=2, min_samples_split=2, random_state=0)

#estimator = nb of free in forest, nbjobs = parallel calcul using cpu

#scores = cross_val_score(clf, X, y)

#scores.mean()    

classif.fit(x_train, y_train.values.ravel())

y_pred_test_rf = classif.predict(x_test)

print("RANDOMFOREST")

print(confusion_matrix(y_test, y_pred_test_rf))

print(f1_score(y_test, y_pred_test_rf))

print(classification_report(y_test, y_pred_test_rf))





# Decision Tree, baby!

classif2 = DecisionTreeClassifier(max_depth=None, min_samples_split=2,random_state=0)

classif2.fit(x_train, y_train.values.ravel())

y_pred_test_clf2 = classif2.predict(x_test)

#scores = cross_val_score(clf, x_train, y_train)

#scores.mean()

print("DECISIONTREE1")

print(confusion_matrix(y_test, y_pred_test_clf2))

print(f1_score(y_test, y_pred_test_clf2))

print(classification_report(y_test, y_pred_test_clf2))



classif2 = DecisionTreeClassifier(max_depth=None, min_samples_split=2,random_state=15)

classif2.fit(x_train, y_train.values.ravel())

y_pred_test_clf2 = classif2.predict(x_test)

#scores = cross_val_score(clf, x_train, y_train)

#scores.mean()

print("DECISIONTREE2")

print(confusion_matrix(y_test, y_pred_test_clf2))

print(f1_score(y_test, y_pred_test_clf2))

print(classification_report(y_test, y_pred_test_clf2))







# Extra Trees 4 social good, peepz!

classif3 = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)

#scores = cross_val_score(clf, X, y)

#scores.mean() > 0.999

classif3.fit(x_train, y_train.values.ravel())

y_pred_test_clf2 = classif3.predict(x_test)

#scores = cross_val_score(clf, x_train, y_train)

#scores.mean()

print("DECISIONTREE3")

print(confusion_matrix(y_test, y_pred_test_clf2))

print(f1_score(y_test, y_pred_test_clf2))

print(classification_report(y_test, y_pred_test_clf2))





print ("OVERSAMPLING")



#substract the min of the column divide by the max  for the whole column

#also apply on test+validation

#try without normalizing at all, nor stabilize



logi = LogisticRegression(class_weight='balanced')

mdl = logi.fit(X_train, Y_train.values.ravel())

predictions2 = logi.predict(X_test)

print("ACCURACY: ",accuracy_score(Y_test, predictions2))

print ("CMATRIX: ")

print (confusion_matrix(Y_test, predictions2))

print (classification_report(Y_test, predictions2))

print("LOGICREGRESS2")





svc = SVC(C=1, kernel='linear')

svc.fit (X_train,Y_train.values.ravel())

ypredsvc = svc.predict (X_test)

scores = cross_val_score(svc, x_train_res, y_train_res)

print ("MYSCORE")

print (scores)

print(confusion_matrix(Y_test, ypredsvc))

print(f1_score(Y_test, ypredsvc))

print("SVM")



# Random Forest stories, mate!



classif = RandomForestClassifier(n_estimators=100, n_jobs=2, min_samples_split=2, random_state=0)

#estimator = nb of free in forest, nbjobs = parallel calcul using cpu

#scores = cross_val_score(clf, X, y)

#scores.mean()    

classif.fit(X_train, Y_train.values.ravel())

Y_pred_test_rf = classif.predict(X_test)

print("RANDOMFOREST")

print(confusion_matrix(Y_test, Y_pred_test_rf))

print(f1_score(Y_test, Y_pred_test_rf))





# Decision Tree, baby!

classif2 = DecisionTreeClassifier(max_depth=None, min_samples_split=2,random_state=0)

classif2.fit(X_train, Y_train.values.ravel())

Y_pred_test_clf2 = classif2.predict(X_test)

#scores = cross_val_score(clf, x_train, y_train)

#scores.mean()

print("DECISIONTREE")

print(confusion_matrix(Y_test, Y_pred_test_clf2))

print(f1_score(Y_test, Y_pred_test_clf2))



# Neural network, captain!

lr = LogisticRegression(C = 5, penalty = 'l1')

lr.fit(X_train, Y_train.values.ravel())

Y_pred_test_nn = lr.predict(X_test)

print("NEURALNETWORK1")

print(confusion_matrix(Y_test, Y_pred_test_nn))

print(f1_score(Y_test, Y_pred_test_nn))





# Extra Trees 4 social good, peepz!

classif3 = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)

#scores = cross_val_score(clf, X, y)

#scores.mean() > 0.999

'''The main parameters to adjust when using these methods is n_estimators and max_features. 

The larger n_estimators the better, but also the longer it  will take to compute. 

max_feat is the size of the random subsets of features to consider when splitting a node. 

lower = greater the reduction of variance, but also the greater the increase in bias. 

Empirical good default values are max_features=n_features for regression problems, 

and max_features=sqrt(n_features) for classification tasks (where n_features is the number of features 

in the data). Good results are often achieved when setting max_depth=None in combination with min_samples_split=1 

(i.e., when fully developing the trees). The best parameter values should always be cross-validated. 

In addition, note that in random forests, bootstrap samples are used by default (bootstrap=True) 

while the default strategy for extra-trees is to use the whole dataset (bootstrap=False). When using 

bootstrap sampling the generalization accuracy can be estimated on the left out or out-of-bag samples. 

This can be enabled by setting oob_score=True.'''
tn, fp, fn,tp = confusion_matrix(predictions,y_test).ravel() 

Sensitivity=tp/float((tp+fn))#Sensitivity 

print ("SENS",Sensitivity)



Specificity=tn/float((tn+fp))#Specificity 

print ("SPEC",Specificity)



Accuracy= accuracy_score(predictions,y_test, normalize=True, sample_weight=None)

print ("ACC",Accuracy)



tn, fp, fn,tp = confusion_matrix(predictions2,Y_test).ravel() 

Sensitivity=tp/float((tp+fn))#Sensitivity 

print ("SENS",Sensitivity)



Specificity=tn/float((tn+fp))#Specificity 

print ("SPEC",Specificity)



Accuracy= accuracy_score(predictions2,Y_test, normalize=True, sample_weight=None)

print ("ACC",Accuracy)