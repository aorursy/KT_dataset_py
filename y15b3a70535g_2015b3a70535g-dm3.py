import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix,classification_report





malware=pd.read_csv('../input/opcode_frequency_malware.csv')

benign=pd.read_csv('../input/opcode_frequency_benign.csv')



# Any results you write to the current directory are saved as output.
#see if there any null data points



malware.isnull().sum().sum()

benign.isnull().sum().sum()
malware.head()
malware['1809']=1

benign['1809']=0
#concatenate benign and malware sets after assigning values to benign and malware



newdata=[benign,malware]

newdata = pd.concat(newdata) #this is our training data set

#we also observe that there are no categorical variables



#drop duplicate tuples and drop filename column



newdata = newdata.drop_duplicates()

newdata = newdata.drop(columns=['FileName'], axis=1)



newdata['1809'].value_counts()



#no duplicate rows found.
#dividing train data into train, test data . 70-30

X=newdata.drop(columns=['1809'],axis=1)

y=newdata['1809']



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,train_size=0.7, random_state=42)



#Classification algorithms

from sklearn.preprocessing import MinMaxScaler





#1. KNN. Scaling needed since it depends upon distance

sc=MinMaxScaler()

data_scaled = sc.fit_transform(X)

#convert to dataframe

data_scaled = pd.DataFrame(data_scaled, columns=X.columns)

scaled_Xtrain, scaled_Xtest, scaled_Ytrain, scaled_Ytest = train_test_split(data_scaled,y,test_size=0.3,random_state=42)



neighbors=[3,5,7,8,10]



for i in neighbors:

    clf = KNeighborsClassifier(n_neighbors=i)

    clf.fit(scaled_Xtrain, scaled_Ytrain)

    print(accuracy_score(clf.predict(scaled_Xtest), scaled_Ytest))



#finally on the entire training set

knn_final=KNeighborsClassifier(n_neighbors=3)

knn_final.fit(data_scaled,y)

print(accuracy_score(knn_final.predict(data_scaled),y))

print(confusion_matrix(knn_final.predict(X), y))

print(classification_report(knn_final.predict(X), y))

#2. Decision Tree



max_depth=[3,5,10,12,16,19,20,21]



for i in max_depth:

    clf = DecisionTreeClassifier(max_depth=i)

    clf.fit(X_train, y_train)

    print(accuracy_score(clf.predict(X_test), y_test))

    

    

#on entire train dataset

dt_final=DecisionTreeClassifier(max_depth=12)

dt_final.fit(X,y)

print(accuracy_score(dt_final.predict(X),y))

print(confusion_matrix(dt_final.predict(X), y))

print(classification_report(dt_final.predict(X), y))
#3. Random Forest



#We have assumed no. of estimators=200 and then chosen max_depth same as decision tree



max_depth=[3,5,10,12,16,19,20,21]



for i in max_depth:

    clf = RandomForestClassifier(max_depth=i, n_estimators=200)

    clf.fit(X_train, y_train)

    print(accuracy_score(clf.predict(X_test), y_test))

    

rf_final=RandomForestClassifier(max_depth=19,n_estimators=200)

rf_final.fit(X,y)

print(accuracy_score(rf_final.predict(X),y))

print(confusion_matrix(rf_final.predict(X), y))

print(classification_report(rf_final.predict(X), y))
#4. PCA 



#We will try with 5,10,15 and 20 components



from sklearn import decomposition

pca = decomposition.PCA(n_components=5)

pca1 = decomposition.PCA(n_components=10)



pca2 = decomposition.PCA(n_components=15)



pca3 = decomposition.PCA(n_components=20)





X1 = pca.fit_transform(newdata)

X2 = pca1.fit_transform(newdata)



X3 = pca2.fit_transform(newdata)



X4 = pca3.fit_transform(newdata)



dfpca1 = pd.DataFrame(data=X1, columns=np.arange(5))

dfpca2 = pd.DataFrame(data=X2, columns=np.arange(10))

dfpca3 = pd.DataFrame(data=X3, columns=np.arange(15))

dfpca4 = pd.DataFrame(data=X4, columns=np.arange(20))



dfpca4.head()
#we'll try our standard classifiers (only DT, RF) on the above new datasets. 



X_train1, X_test1, y_train1, y_test1 = train_test_split(dfpca1, y, test_size=0.3, random_state=42)



X_train2, X_test2, y_train2, y_test2 = train_test_split(dfpca2, y, test_size=0.3, random_state=42)



X_train3, X_test3, y_train3, y_test3 = train_test_split(dfpca3, y, test_size=0.3, random_state=42)



X_train4, X_test4, y_train4, y_test4 = train_test_split(dfpca4, y, test_size=0.3, random_state=42)
max_depth=[3,5,10,12,16,19,20,21]



for i in max_depth:

    clf = RandomForestClassifier(max_depth=i, n_estimators=200)

    clf.fit(X_train1, y_train1)

    print(accuracy_score(clf.predict(X_test1), y_test1))
max_depth=[3,5,10,12,16,19,20,21]



for i in max_depth:

    clf = RandomForestClassifier(max_depth=i, n_estimators=200)

    clf.fit(X_train2, y_train2)

    print(accuracy_score(clf.predict(X_test2), y_test2))
max_depth=[3,5,10,12,16,19,20,21]



for i in max_depth:

    clf = RandomForestClassifier(max_depth=i, n_estimators=200)

    clf.fit(X_train3, y_train3)

    print(accuracy_score(clf.predict(X_test3), y_test3))
max_depth=[3,5,10,12,16,19,20,21]



for i in max_depth:

    clf = RandomForestClassifier(max_depth=i, n_estimators=200)

    clf.fit(X_train4, y_train4)

    print(accuracy_score(clf.predict(X_test4), y_test4))
max_depth=[3,5,10,12,16,19,20,21]



for i in max_depth:

    clf = DecisionTreeClassifier(max_depth=i)

    clf.fit(X_train1, y_train1)

    print(accuracy_score(clf.predict(X_test1), y_test1))
max_depth=[3,5,10,12,16,19,20,21]



for i in max_depth:

    clf = DecisionTreeClassifier(max_depth=i)

    clf.fit(X_train2, y_train2)

    print(accuracy_score(clf.predict(X_test2), y_test2))
max_depth=[3,5,10,12,16,19,20,21]



for i in max_depth:

    clf = DecisionTreeClassifier(max_depth=i)

    clf.fit(X_train3, y_train3)

    print(accuracy_score(clf.predict(X_test3), y_test3))
max_depth=[3,5,10,12,16,19,20,21]



for i in max_depth:

    clf = DecisionTreeClassifier(max_depth=i)

    clf.fit(X_train4, y_train4)

    print(accuracy_score(clf.predict(X_test4), y_test4))
#The best RF Model was on PCA=15 components,max_depth=20 

#hence I'm reporting confusion matrix for that case wrt PCA





rf_pca=RandomForestClassifier(max_depth=20,n_estimators=200)

rf_pca.fit(X3,y)

print(accuracy_score(rf_final.predict(X),y))

print(confusion_matrix(rf_final.predict(X), y))

print(classification_report(rf_final.predict(X), y))
#Testing on best RF model - depth 19,n_estimators=200 (later tried with 150 and that gave better).



testdata = pd.read_csv("../input/Test_data.csv")

#The next line of code is to accommodate for download link

f = testdata['FileName']

#back to usual

testdata = testdata.drop(columns=['FileName'],axis=1)
testdata.head()

testdata=testdata.dropna(axis=1)

testdata.head()
pcatest=decomposition.PCA(n_components=15)

Xtest=pcatest.fit_transform(testdata)

datafinal=pd.DataFrame(data=Xtest,columns=np.arange(15))

datafinal.head()
#The final model



clf = RandomForestClassifier(max_depth=19, n_estimators=150)

clf.fit(X,y)

pred = clf.predict(testdata)

pred



#score at depth=16, est=100 was 0.98342

#score at depth=20, est=150 was 0.98552

#score at depth=21, est=200 was 0.98308

#score at depth=19, est=150 was 0.98598. best out of all on testdata
np.savetxt("ans.csv",pred,header="Class")



#Does not have filename. I embedded that column from train dataset separately and then ran on kaggle



#In the following code snippet(which is not there in my moodle submission I changed pred to dataframe pred along with filename to accommodate for the download csv code,

#though I managed submitting it on kaggle without it as it was already submitted in a csv file.. 

#Please consider the same



final = pd.DataFrame()

final['Filename']=f

final['Class']=pred

final.head()
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



create_download_link(final)