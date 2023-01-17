import numpy as np # 수학 연산 수행을 위한 모듈

import pandas as pd # 데이터 처리를 위한 모듈

import seaborn as sns # 데이터 시각화 모듈

import matplotlib.pyplot as plt # 데이터 시각화 모듈 
# 어떤 파일이 있는지 표시하기

from subprocess import check_output



print(check_output(["ls", "../input/male-female"]).decode("utf8"))
# CSV 파일 읽어오기

data_frame = pd.read_csv("../input/male-female/male_female.csv")
print(data_frame.info())
data_frame.head(5)
for col in data_frame.columns: 

    print(col) 
data_frame.hist(edgecolor='black', linewidth=1.2)

fig = plt.gcf()

fig.set_size_inches(12,10)

plt.show()
# 읽어온 데이터 표시하기

cl = data_frame['Sex'].unique()



col = ['orange', 'blue', 'red', 'yellow', 'black', 'brown']



fig = data_frame[data_frame['Sex'] == cl[0]].plot(kind='scatter', x='FeetSize', y='Height', color=col[0], label=cl[0])



for i in range(len(cl)-1):

    data_frame[data_frame['Sex'] == cl[i+1]].plot(kind='scatter', x='FeetSize', y='Height', color=col[i+1], label=cl[i+1], ax=fig)



fig.set_xlabel('FeetSize')

fig.set_ylabel('Height')

fig.set_title('FeetSize' + " vs. " + 'Height')

fig=plt.gcf()

fig.set_size_inches(10, 6)

plt.show()
f, sub = plt.subplots(1, 1,figsize=(8,6))

sns.boxplot(x=data_frame['Sex'],y=data_frame['FeetSize'], ax=sub)

sub.set(xlabel='Sex', ylabel='FeetSize')
plt.figure(figsize=(8,6))

plt.subplot(1,1,1)

sns.violinplot(x='Sex',y='Height',data=data_frame)

from mpl_toolkits.mplot3d import Axes3D



fig=plt.figure(figsize=(12,8))



ax=fig.add_subplot(1,1,1, projection="3d")

ax.scatter(data_frame['Height'],data_frame['Weight'],data_frame['FeetSize'],c="blue",alpha=.5)

ax.set(xlabel='Height',ylabel='Weight',zlabel='FeetSize')
plt.figure(figsize=(12,8)) 

sns.heatmap(data_frame.corr(),annot=True,cmap='cubehelix_r') 

plt.show()
from sklearn.model_selection import train_test_split



train, test = train_test_split(data_frame, test_size = 0.2)



# train=70% and test=30%

print(train.shape)

print(test.shape)
# 학습용 문제, 학습용 정답

train_X = train[['Height','FeetSize']] # 키와 발크기만 선택

train_y = train.Sex # 정답 선택



# 테스트용 문제, 테스트용 정답

test_X = test[['Height','FeetSize']] # taking test data features

test_y = test.Sex   #output value of test data
# 다양한 분류 알고리즘 패키지를 임포트함.

from sklearn.linear_model import LogisticRegression  # Logistic Regression 알고리즘

#from sklearn.cross_validation import train_test_split # 데이타 쪼개주는 모듈 

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier  # for K nearest neighbours

from sklearn import svm  #for Support Vector Machine (SVM) Algorithm

from sklearn import metrics #for checking the model accuracy

from sklearn.tree import DecisionTreeClassifier #for using Decision Tree Algoithm
gildong = svm.SVC()

gildong.fit(train_X,train_y) # 가르친 후

prediction = gildong.predict(test_X) # 얼마나 맞히는지 테스트



rate1 = metrics.accuracy_score(prediction,test_y) * 100

print('인식률: {0:.1f}'.format(rate1))
cheolsu = LogisticRegression()

cheolsu.fit(train_X,train_y)

prediction = cheolsu.predict(test_X)



rate2 = metrics.accuracy_score(prediction,test_y) * 100

print('인식률: {0:.1f}'.format(rate2))
youngja = DecisionTreeClassifier()

youngja.fit(train_X,train_y)

prediction = youngja.predict(test_X)



rate3 = metrics.accuracy_score(prediction,test_y) * 100

print('인식률: {0:.1f}'.format(rate3))

minsu = KNeighborsClassifier(n_neighbors=3) #this examines 3 neighbours for putting the new data into a class

minsu.fit(train_X,train_y)

prediction = minsu.predict(test_X)



rate4 = metrics.accuracy_score(prediction,test_y) * 100

print('인식률: {0:.1f}'.format(rate4))
plt.plot(['SVM','Logistic','D-Tree','K-NN'], [rate1, rate2, rate3, rate4])
train_X = train[['Height','FeetSize','Weight']] # 키와 발크기뿐만 아니라 몸무게도 추가

train_y = train.Sex # 정답 선택



test_X = test[['Height','FeetSize','Weight']] # taking test data features

test_y = test.Sex   #output value of test data
def run_4_classifiers(a, b, c, d):

    tmp = svm.SVC() # 애기 

    tmp.fit(a,b) # 가르친 후

    prediction = tmp.predict(c) # 테스트

    rate1 = metrics.accuracy_score(prediction,test_y) * 100



    tmp = LogisticRegression()

    tmp.fit(a,b)

    prediction = tmp.predict(c)

    rate2 = metrics.accuracy_score(prediction,test_y) * 100



    tmp = DecisionTreeClassifier()

    tmp.fit(a,b)

    prediction = tmp.predict(c)

    rate3 = metrics.accuracy_score(prediction,test_y) * 100



    tmp = KNeighborsClassifier(n_neighbors=3) #this examines 3 neighbours for putting the new data into a class

    tmp.fit(a,b)

    prediction = tmp.predict(c)

    rate4 = metrics.accuracy_score(prediction,test_y) * 100

    

    plt.plot(['SVM','Logistic','D-Tree','K-NN'], [rate1, rate2, rate3, rate4])

    print('인식률: {0:.2f}, {1:.2f}, {2:.2f}, {3:.2f}'.format(rate1, rate2, rate3, rate4))

    

    

run_4_classifiers(train_X, train_y, test_X, test_y)
