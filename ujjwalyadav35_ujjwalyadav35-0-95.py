import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import sys
import pickle
from sklearn import preprocessing
from time import time
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


import sys
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

sys.path.append('/kaggle/input/enronn/')
from feature_format import featureFormat
from feature_format import targetFeatureSplit

original = "/kaggle/input/enronn/final_project_dataset.pkl"
destination = "final_project_dataset_unix.pkl"

content = ''
outsize = 0
with open(original, 'rb') as infile:
    content = infile.read()
with open(destination, 'wb') as output:
    for line in content.splitlines():
        outsize += len(line) + 1
        output.write(line + str.encode('\n'))
df = pickle.load(open("final_project_dataset_unix.pkl", "rb") )
df.pop('TOTAL')
df
df2=pd.DataFrame(df)
df2
def featureFormat( dictionary, features, remove_NaN=True, remove_all_zeroes=True, remove_any_zeroes=False, sort_keys = False):
    return_list = []
    if isinstance(sort_keys, str):
        import pickle
        keys = pickle.load(open(sort_keys, "rb"))
    elif sort_keys:
        keys = sorted(dictionary.keys())
    else:
        keys = dictionary.keys()

    for key in keys:
        tmp_list = []
        for feature in features:
            try:
                dictionary[key][feature]
            except KeyError:
                print ("error: key ", feature, " not present")
                return
            value = dictionary[key][feature]
            if value=="NaN" and remove_NaN:
                value = 0
            tmp_list.append( float(value) )

        # Logic for deciding whether or not to add the data point.
        append = True
        # exclude 'poi' class as criteria.
        if features[0] == 'poi':
            test_list = tmp_list[1:]
        else:
            test_list = tmp_list
        ### if all features are zero and you want to remove
        ### data points that are all zero, do that here
        if remove_all_zeroes:
            append = False
            for item in test_list:
                if item != 0 and item != "NaN":
                    append = True
                    break
        ### if any features for a given data point are zero
        ### and you want to remove data points with any zeroes,
        ### handle that here
        if remove_any_zeroes:
            if 0 in test_list or "NaN" in test_list:
                append = False
        ### Append the data point if flagged for addition.
        if append:
            return_list.append( np.array(tmp_list) )

    return np.array(return_list)
def targetFeatureSplit( data ):
    target = []
    features = []
    for item in data:
        target.append( item[0] )
        features.append( item[1:] )

    return target, features
features_list = ['poi','salary','total_payments'] 
pst = featureFormat(df, ['poi','salary','total_payments'])
# featureFormat()
print("The number of people in the dataset:",len(df))


for i in range(len(pst)):
    if pst[i][1]<1000000:
        if pst[i][0]==True:
            plt.scatter(pst[i][1],pst[i][2],color = 'b')
        else:
            plt.scatter(pst[i][1],pst[i][2],color = 'r')


plt.ylabel('tot_payments')
plt.xlabel('salary')   
plt.show()
#PLotting relationship between different features
fig,a=plt.subplots(4,2,squeeze=False,figsize=(15,12))
psd=featureFormat(df,["poi","salary","deferral_payments"])
for j in range(len(psd)):
    if psd[j][0]==True:
        a[0][0].scatter(psd[j][1],psd[j][2],color='r')
    else:
        a[0][0].scatter(psd[j][1],psd[j][2],color='g')
ptl=featureFormat(df,["poi","total_payments","loan_advances"])
for j in range(len(ptl)):
    if ptl[j][0]==True:
        a[0][1].scatter(ptl[j][1],ptl[j][2],color='r')
    else:
        a[0][1].scatter(ptl[j][1],ptl[j][2],color='g')
pte=featureFormat(df,["poi","total_stock_value","exercised_stock_options"])
for j in range(len(pte)):
    if pte[j][0]==True:
        a[1][0].scatter(pte[j][1],pte[j][2],color='r')
    else:
        a[1][0].scatter(pte[j][1],pte[j][2],color='g')
psld=featureFormat(df,["poi","salary","loan_advances","deferral_payments"])
for j in range(len(psld)):
    if psld[j][0]==True:
        a[1][1].scatter(psld[j][1],psld[j][2],color='r')
    else:
        a[1][1].scatter(psld[j][1],psld[j][2],color='g')
ptr=featureFormat(df,["poi","total_stock_value","restricted_stock"])
for j in range(len(ptr)):
    if ptr[j][0]==True:
        a[2][0].scatter(ptr[j][1],ptr[j][2],color='r')
    else:
        a[2][0].scatter(ptr[j][1],ptr[j][2],color='g')     
        
per=featureFormat(df,["poi","exercised_stock_options","restricted_stock"])
for j in range(len(per)):
    if per[j][0]==True:
        a[2][1].scatter(per[j][1],per[j][2],color='r')
    else:
        a[2][1].scatter(per[j][1],per[j][2],color='g') 
        
psb=featureFormat(df,["poi","salary","bonus"])
for j in range(len(psb)):
    if psb[j][0]==True:
        a[3][0].scatter(psb[j][1],psb[j][2],color='r')
    else:
        a[3][0].scatter(psb[j][1],psb[j][2],color='g') 
        
abc5 =  featureFormat(df,['poi','from_this_person_to_poi','from_poi_to_this_person'])
for key in range(len(abc5)):
    if abc5[key][0] == True:
         a[3][1].scatter(abc5[key][1],abc5[key][2],color = 'r')
    else:
        a[3][1].scatter(abc5[key][1],abc5[key][2],color = 'g')

plt.show()
psb
#Defining new features

def dict_to_list(key,normalizer):
    feature_list=[]

    for i in df:
        if df[i][key]=="NaN" or df[i][normalizer]=="NaN":
            feature_list.append(0.)
        elif df[i][key]>=0:
            feature_list.append(float(df[i][key])/float(df[i][normalizer]))
    return feature_list

fraction_from_poi_email=dict_to_list("from_poi_to_this_person","to_messages")
fraction_to_poi_email=dict_to_list("from_this_person_to_poi","from_messages")
p = 0
for i in df:
    df[i]["fraction_from_poi_email"]=fraction_from_poi_email[p]
    df[i]["fraction_to_poi_email"]=fraction_to_poi_email[p]
    p=p+1
#Training and testing data using different methods

feature_list = ['poi','shared_receipt_with_poi','fraction_from_poi_email','fraction_to_poi_email',"deferral_payments"]
df2 = featureFormat(df, feature_list)
labels, features = targetFeatureSplit(df2)
X_train, X_test, Y_train, Y_test = train_test_split(features, labels,test_size=0.2,random_state=42)
# Using Gaussian Naive Bayes

gb=GaussianNB()
gb.fit(X_train,Y_train)
print("Accuracy for GaussianNB: ",accuracy_score(Y_test, gb.predict(X_test)))
# Using random forest regression

rfr=RandomForestClassifier(max_depth=10, criterion = 'entropy')
rfr.fit(X_train,Y_train)
print("Accuracy for RandomForestClassifier:",accuracy_score(Y_test, rfr.predict(X_test)))
# Using K-nearest neighbour algo

import sklearn.metrics as metrics
from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(n_neighbors=7)
KNN.fit(X_train,Y_train)
pred = KNN.predict(X_test)
acc = metrics.accuracy_score(pred, Y_test)
print("Accuracy by KNN classifier: ",acc)
# Using Decision Tree Classifier

from sklearn import tree
DTC = tree.DecisionTreeClassifier()
DTC = DTC.fit(X_train, Y_train)
pred2 = DTC.predict(X_test)
acc2 = metrics.accuracy_score(pred2,Y_test)
print("Accuracy using Decision Tree classifier: ",acc2)
pickle.dump(KNN, open("my_classifier.pkl", "wb") )
pickle.dump(df, open("df.pkl", "wb") )
pickle.dump(feature_list, open("my_feature_list.pkl", "wb") )
