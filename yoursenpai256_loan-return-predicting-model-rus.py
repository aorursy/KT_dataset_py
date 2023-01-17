import pandas as pd, numpy as np,collections as cl, matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import f1_score

from sklearn.utils.multiclass import unique_labels
data = pd.read_csv('../input/hmeq.csv')
data.dropna(how='any', inplace=True)
for column in data:

    if column not in ['BAD','REASON','JOB']:         

        mean=data[column].mean()       

        std=data[column].std()       

        data[column]=[(x-mean)/std for x in data[column]]

        data[column]=pd.cut(data[column],20)    
le = LabelEncoder()

for column in data.columns:

    encc = le.fit(data[column].astype(str))  

    tr = le.transform(data[column].astype(str))

    data[column]=tr
data1 = data.copy()

bad=data['BAD']

del data['BAD']
x_train, x_test, y_train, y_test = train_test_split(data, bad, test_size=0.33, random_state=42)   
clrTree = DecisionTreeClassifier(max_depth = 3)

clrTree = clrTree.fit(x_train, y_train)

pred = list(clrTree.predict(x_test))   

precision=precision_score(y_test,pred)

recall=recall_score(y_test,pred)

f1 = f1_score(y_test,pred)

print("Accuracy decisiontree = {0:.5f}".format(clrTree.score(x_test,y_test)))

print("Precision= {0:.5f}\nRecall = {1:.5f}\nF1={2:.5f}\n".format(precision,recall,f1))
neigh = KNeighborsClassifier(n_neighbors=5)

neigh.fit(x_train, y_train) 

pred = list(neigh.predict(x_test))  

precision=precision_score(y_test,pred)

recall=recall_score(y_test,pred)

f1 = f1_score(y_test,pred)

print("Accuracy KNeighborsClassifier = {0:.5f}".format(neigh.score(x_test,y_test)))

print("Precision= {0:.5f}\nRecall = {1:.5f}\nF1={2:.5f}\n".format(precision,recall,f1))
rf = RandomForestClassifier(max_depth=10)

rf.fit(x_train, y_train)

pred = list(rf.predict(x_test))  

precision=precision_score(y_test,pred)

recall=recall_score(y_test,pred)

f1 = f1_score(y_test,pred)

print("Accuracy RandomForestClassifier = {0:.5f}".format(rf.score(x_test,y_test)))

print("Precision= {0:.5f}\nRecall = {1:.5f}\nF1={2:.5f}\n".format(precision,recall,f1))
zero_index=data1[data1["BAD"]==0].index

one_index=data1[data1["BAD"]==1].index

data1['BAD'].loc[zero_index] = 1

data1['BAD'].loc[one_index] = 0

bad=data1['BAD']

del data1['BAD']
x_train, x_test, y_train, y_test = train_test_split(data1, bad, test_size=0.33, random_state=42)   



clrTree = DecisionTreeClassifier(max_depth = 3)

clrTree = clrTree.fit(x_train, y_train)

pred = list(clrTree.predict(x_test))   

precision=precision_score(y_test,pred)

recall=recall_score(y_test,pred)

f1 = f1_score(y_test,pred)

print("Accuracy decisiontree = {0:.5f}".format(clrTree.score(x_test,y_test)))

print("Precision= {0:.5f}\nRecall = {1:.5f}\nF1={2:.5f}\n".format(precision,recall,f1))



neigh = KNeighborsClassifier(n_neighbors=5)

neigh.fit(x_train, y_train) 

pred = list(neigh.predict(x_test))  

precision=precision_score(y_test,pred)

recall=recall_score(y_test,pred)

f1 = f1_score(y_test,pred)

print("Accuracy KNeighborsClassifier = {0:.5f}".format(neigh.score(x_test,y_test)))

print("Precision= {0:.5f}\nRecall = {1:.5f}\nF1={2:.5f}\n".format(precision,recall,f1))



rf = RandomForestClassifier(max_depth=10)

rf.fit(x_train, y_train)

pred = list(rf.predict(x_test))  

precision=precision_score(y_test,pred)

recall=recall_score(y_test,pred)

f1 = f1_score(y_test,pred)

print("Accuracy RandomForestClassifier = {0:.5f}".format(rf.score(x_test,y_test)))

print("Precision= {0:.5f}\nRecall = {1:.5f}\nF1={2:.5f}\n".format(precision,recall,f1))