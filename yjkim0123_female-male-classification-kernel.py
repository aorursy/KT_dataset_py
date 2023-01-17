# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
iris = pd.read_csv("../input/person.csv") #데이터 파일을 읽어서 2차원 테이블 형식의 데이터 프레임을 만들어줌.
iris.head(10) # 데이터 프레임에 들어있는 10개 데이터 보여줌.
iris.info() # 데이터에 비어있는(널) 값이 있는지 확인할 수 있음.
iris.drop('Id',axis=1,inplace=True) 

# 불필요한 Id 컬럼 삭제

# axis=1 : 컬럼을 의미

# inplace=1 : 삭제한 후 데이터 프레임에 반영
fig = iris[iris.Sex==0].plot(kind='scatter',x='Height',y='Weight',color='orange', label='Female')

iris[iris.Sex==1].plot(kind='scatter',x='Height',y='Weight',color='blue', label='Male', ax=fig)

fig.set_xlabel("Height")

fig.set_ylabel("Weight")

fig.set_title("Height VS Weight")

fig=plt.gcf()

fig.set_size_inches(10,6)

plt.show()

fig = iris[iris.Sex==0].plot(kind='scatter',x='Height',y='FeetSize',color='orange', label='Female')

iris[iris.Sex==1].plot(kind='scatter',x='Height',y='FeetSize',color='blue', label='Male', ax=fig)

fig.set_xlabel("Height")

fig.set_ylabel("Weight")

fig.set_title("Height VS Weight")

fig=plt.gcf()

fig.set_size_inches(10,6)

plt.show()


fig = iris[iris.Species=='Iris-setosa'].plot(kind='scatter',x='PetalLengthCm',y='PetalWidthCm',color='orange', label='Setosa')

iris[iris.Species=='Iris-versicolor'].plot(kind='scatter',x='PetalLengthCm',y='PetalWidthCm',color='blue', label='versicolor', ax=fig)

iris[iris.Species=='Iris-virginica'].plot(kind='scatter',x='PetalLengthCm',y='PetalWidthCm',color='green', label='virginica', ax=fig)

fig.set_xlabel("Sepal Length")

fig.set_ylabel("Sepal Width")

fig.set_title("Sepal Length VS Width")

fig=plt.gcf()

fig.set_size_inches(10,6)

plt.show()
iris.hist(edgecolor='black', linewidth=1.2)

fig = plt.gcf()

fig.set_size_inches(12,6)

plt.show()
plt.figure(figsize=(15,10))

plt.subplot(2,2,1)

sns.violinplot(x='Sex',y='Height',data=iris)

plt.subplot(2,2,2)

sns.violinplot(x='Sex',y='FeetSize',data=iris)

plt.subplot(2,2,3)

sns.violinplot(x='Sex',y='Weight',data=iris)

plt.subplot(2,2,4)

sns.violinplot(x='Sex',y='Year',data=iris)
# 다양한 분류 알고리즘 패키지를 임포트함.

from sklearn.linear_model import LogisticRegression  # Logistic Regression 알고리즘

#from sklearn.cross_validation import train_test_split # 데이타 쪼개주는 모듈 

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier  # for K nearest neighbours

from sklearn import svm  #for Support Vector Machine (SVM) Algorithm

from sklearn import metrics #for checking the model accuracy

from sklearn.tree import DecisionTreeClassifier #for using Decision Tree Algoithm
iris.shape # 데이터 프레임 모양(shape)

iris.head(10)
def draw_heatmap(df):

    plt.figure(figsize=(7,4)) 

    sns.heatmap(df.corr(),annot=True,cmap='cubehelix_r') #draws  heatmap with input as the correlation matrix calculted by(iris.corr())

    plt.show()



draw_heatmap(iris)
def split(df, p):

    a, b = train_test_split(df, test_size = p)# in this our main data is split into train and test

    # the attribute test_size=0.3 splits the data into 70% and 30% ratio. train=70% and test=30%

    print(a.shape)

    print(b.shape)

    return a, b



train, test = split(iris, 0.3)
train_X = train[['Height','FeetSize']]# taking the training data features

train_y=train.Sex# output of our training data



test_X= test[['Height','FeetSize']] # taking test data features

test_y =test.Sex   #output value of test data
train_X.head(2)
test_X.head(2)
train_y.head()  ##output of the training data
import warnings  

warnings.filterwarnings('ignore')
model = svm.SVC() # 애기 

model.fit(train_X,train_y) # 가르친 후

prediction=model.predict(test_X) # 테스트



print('정확도:',metrics.accuracy_score(prediction,test_y) * 100)#now we check the accuracy of the algorithm. 

#we pass the predicted output by the model and the actual output
model = LogisticRegression()

model.fit(train_X,train_y)

prediction=model.predict(test_X)

print('인식률:',metrics.accuracy_score(prediction,test_y))
model=DecisionTreeClassifier()

model.fit(train_X,train_y)

prediction=model.predict(test_X)

print('The accuracy of the Decision Tree is',metrics.accuracy_score(prediction,test_y))
model=KNeighborsClassifier(n_neighbors=3) #this examines 3 neighbours for putting the new data into a class

model.fit(train_X,train_y)

prediction=model.predict(test_X)

print('The accuracy of the KNN is',metrics.accuracy_score(prediction,test_y))
petal=iris[['PetalLengthCm','PetalWidthCm','Species']]

sepal=iris[['SepalLengthCm','SepalWidthCm','Species']]
train_p,test_p=train_test_split(petal,test_size=0.3,random_state=0)  #petals

train_x_p=train_p[['PetalWidthCm','PetalLengthCm']]

train_y_p=train_p.Species

test_x_p=test_p[['PetalWidthCm','PetalLengthCm']]

test_y_p=test_p.Species





train_s,test_s=train_test_split(sepal,test_size=0.3,random_state=0)  #Sepal

train_x_s=train_s[['SepalWidthCm','SepalLengthCm']]

train_y_s=train_s.Species

test_x_s=test_s[['SepalWidthCm','SepalLengthCm']]

test_y_s=test_s.Species
model=svm.SVC()

model.fit(train_x_p,train_y_p) 

prediction=model.predict(test_x_p) 

print('The accuracy of the SVM using Petals is:',metrics.accuracy_score(prediction,test_y_p))



model=svm.SVC()

model.fit(train_x_s,train_y_s) 

prediction=model.predict(test_x_s) 

print('The accuracy of the SVM using Sepal is:',metrics.accuracy_score(prediction,test_y_s))
model = LogisticRegression()

model.fit(train_x_p,train_y_p) 

prediction=model.predict(test_x_p) 

print('The accuracy of the Logistic Regression using Petals is:',metrics.accuracy_score(prediction,test_y_p))



model.fit(train_x_s,train_y_s) 

prediction=model.predict(test_x_s) 

print('The accuracy of the Logistic Regression using Sepals is:',metrics.accuracy_score(prediction,test_y_s))
model=DecisionTreeClassifier()

model.fit(train_x_p,train_y_p) 

prediction=model.predict(test_x_p) 

print('The accuracy of the Decision Tree using Petals is:',metrics.accuracy_score(prediction,test_y_p))



model.fit(train_x_s,train_y_s) 

prediction=model.predict(test_x_s) 

print('The accuracy of the Decision Tree using Sepals is:',metrics.accuracy_score(prediction,test_y_s))
model=KNeighborsClassifier(n_neighbors=3) 

model.fit(train_x_p,train_y_p) 

prediction=model.predict(test_x_p) 

print('The accuracy of the KNN using Petals is:',metrics.accuracy_score(prediction,test_y_p))



model.fit(train_x_s,train_y_s) 

prediction=model.predict(test_x_s) 

print('The accuracy of the KNN using Sepals is:',metrics.accuracy_score(prediction,test_y_s))