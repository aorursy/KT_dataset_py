import pandas as pd 



data = pd.read_csv("../input/wisconsin-breast-cancer-cytology/BreastCancer2.csv") # reading data from input file

data= data.drop(["id"],axis=1) # remove useless feature 
data.head() #data features and class
x = data.drop(["class"],axis = 1) # x consist only features

y = data.loc[:,"class"] # y consist only class



print(x.iloc[0:5])

print(y.iloc[0:5])
# AttributeError: 'tuple' object has no attribute 'fit' Therefore we have to transform numpy array

x = data.drop(["class"],axis = 1).values 

y = data.loc[:,"class"].values
from sklearn.model_selection import train_test_split



x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=42) # create a %33 test data from orginal data



print("x train shape:",x_train.shape)

print("x test shape:",x_test.shape)

print("y train shape:",y_train.shape)

print("y test shape:",y_test.shape)



from IPython.display import Image

Image("../input/logistic-shematic/lrs.png")
from IPython.display import Image

Image("../input/function/sigmoid.png")
from sklearn.linear_model import LogisticRegressionCV # Use the sklearn module 

lrc = LogisticRegressionCV()

lrc.fit(x_train,y_train) 

print("logistic score",lrc.score(x_test,y_test)*100)
from IPython.display import Image

Image("../input/knnclass/KNN circle.png")
from  sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5) # K values 

knn.fit(x_train,y_train)

print("KNN score",knn.score(x_test,y_test)*100)
neighbors_list = []



for i in range(1,15):

    knn1 = KNeighborsClassifier(n_neighbors = i)

    knn1.fit(x_train,y_train)

    neighbors_list.append(knn1.score(x_test,y_test))

print(neighbors_list) # find finest k value for classification
from IPython.display import Image

Image("../input/support-vector/SVM.png")
from sklearn.svm import SVC

svc = SVC(random_state=1,gamma=0.22)

svc.fit(x_train,y_train)

print("SVC score",svc.score(x_test,y_test))
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(x_train,y_train)

print("Naive Boyes score",nb.score(x_test,y_test))
from IPython.display import Image

Image("../input/treedesi/decision tree_LI.jpg")
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()

dt.fit(x_train,y_train)

print("Decision Tree Score",dt.score(x_test,y_test))
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators = 10, random_state=1)

rf.fit(x_train,y_train)

print("Random Forest Classification Score",rf.score(x_test,y_test))
from IPython.display import Image

Image("../input/confuison/matrix.png")
from sklearn.metrics import confusion_matrix

y_prediction = rf.predict(x_test) 

cm = confusion_matrix(y_true=y_test,y_pred=y_prediction)#actual value -->y_true

print("Confusion Matrix ",cm)
# Vizualition of Confusion Matrix

import matplotlib.pyplot as plt

import seaborn as sns



f,ax = plt.subplots(figsize=(5,5))

sns.heatmap(cm,annot = True,fmt=".0f")

plt.xlabel("Prediction Value")

plt.ylabel("True Value")

plt.show()
print("Logistic Regression Classification score :",lrc.score(x_test,y_test)*100)

print("KNN score :",knn.score(x_test,y_test)*100)

print("SVC score :",svc.score(x_test,y_test)*100)

print("Naive Boyes score :",nb.score(x_test,y_test)*100)

print("Decision Tree Score :",dt.score(x_test,y_test)*100)

print("Random Forest Classification Score :",rf.score(x_test,y_test)*100)