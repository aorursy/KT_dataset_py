import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
dataset=pd.read_csv("../input/churn-prediction/Churn.csv")
dataset=dataset.drop("customerID",axis=1)
dataset["TotalCharges"]=pd.to_numeric(dataset["TotalCharges"],errors='coerce')
dataset=pd.get_dummies(dataset,drop_first=True)
dataset["TotalCharges"].fillna(1397.4570,inplace=True)
column_list=dataset.columns
data=set(column_list)-set(["Churn_Yes"])
X=dataset[data].values
Y=dataset["Churn_Yes"].values
sc=StandardScaler()
X=sc.fit_transform(X)
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.30,random_state=0)
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import accuracy_score
mylist=[]
for i in range(1,30):
    test=KNeighborsClassifier(n_neighbors=i)
    test.fit(xtrain,ytrain)
    ypred=test.predict(xtest)
    acc=accuracy_score(ytest,ypred)
    mylist.append(acc)
maxx=max(mylist)
print("Maximum Accuracy:",maxx,"at k =",mylist.index(maxx))
from sklearn.linear_model import LogisticRegression 
classifier = LogisticRegression(random_state = 0) 
classifier.fit(xtrain, ytrain)
y_pred = classifier.predict(xtest) 
from sklearn.metrics import accuracy_score 
print ("Accuracy : ", accuracy_score(ytest, y_pred)) 

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics 
clf = DecisionTreeClassifier(criterion="entropy", max_depth=5)
clf = clf.fit(xtrain,ytrain)
y_pred = clf.predict(xtrain)
print("Accuracy:",metrics.accuracy_score(ytrain, y_pred))
from sklearn.svm import SVC
from sklearn import metrics
svclassifier = SVC(kernel='linear')
svclassifier.fit(xtrain, ytrain)
y_pred = svclassifier.predict(xtest)
print(accuracy_score(ytest,y_pred))
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = kmeans.predict(predict_me)
    if prediction[0] == Y[i]:
        correct += 1

print(correct/len(X))
