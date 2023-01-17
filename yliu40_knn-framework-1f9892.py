import numpy as np # 数组常用库
import pandas as pd # 读入csv常用库
import matplotlib.pyplot as plt # 画图常用库
row = 5000
train_csv = pd.read_csv("../input/train.csv")
train_data, train_y = train_csv.values[0:row,1:], train_csv.values[0:row,0]
test_csv = pd.read_csv("../input/test.csv")
test_data = test_csv.values[0:row]
print (train_data.shape, test_data.shape)
import random
num = 5
indices = random.sample(range(0,row),num)
for i in range(num):
    plt.subplot(1,num,i+1)
    plt.imshow(train_data[indices[i]].reshape(28,28))
    plt.xticks([])
    plt.yticks([])
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_data, train_y, test_size = 0.20, random_state = 0)
print (X_train.shape, X_test.shape, y_train.shape, y_test.shape)
acc_list = []
k_num = 8
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
for i in range(k_num):
    classifier = KNeighborsClassifier(n_neighbors = i+1)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    accuracy = accuracy_score(y_test,y_pred)
    acc_list.append(accuracy)
print (acc_list)
fig = plt.figure()
plt.plot(range(1,k_num+1),acc_list)
plt.xlabel('k')
plt.ylabel('accuracy')
print (len(acc_list))
k = 3
classifier = KNeighborsClassifier(n_neighbors = k)
classifier.fit(train_data, train_y)
test_pred = classifier.predict(test_data)
accuracy = accuracy_score(y_test,y_pred)
print (accuracy)
print (test_pred[10])
plt.imshow(test_data[10].reshape(28,28))
plt.show()
print (len(test_pred))
pd.DataFrame({"ImageId": list(range(1,len(y_pred)+1)),"Label": y_pred}).to_csv('Digit_Recogniser_Result.csv', index=False,header=True)
