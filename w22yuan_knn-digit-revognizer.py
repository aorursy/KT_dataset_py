import numpy as np # 数组常用库
import pandas as pd # 读入csv常用库
import matplotlib.pyplot as plt # 画图常用库
orginal_train_data = pd.read_csv('../input/train.csv')
orginal_test_data = pd.read_csv('../input/test.csv')

type(orginal_train_data)
orginal_train_data.head()
orginal_test_data.head()
orginal_x_train = orginal_train_data.values[0:, 1:]
orginal_x_train

orginal_x_train.shape
orginal_y_train = orginal_train_data.values[:, 0]
orginal_y_train
print(orginal_y_train.shape)
print(orginal_test_data.shape)
print(orginal_y_train[123])
plt.imshow(orginal_x_train[123].reshape(28,28))
plt.show()
from sklearn.model_selection import train_test_split

x_train, x_validation, y_train, y_validation = train_test_split(orginal_x_train, orginal_y_train, test_size =0.2, random_state = 0)
print("x_train_shape: " +  str(x_train.shape))
print("x_validation_shape: " + str(x_validation.shape))
print("y_train_shape: " + str(y_train.shape))
print("y_validation_shape: " + str(y_validation.shape))

import time
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

ans_k = 0
score = []

for k in range(1,8):
    print("k = " + str(k) + " begin") 
    start = time.time()
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    y_predict = knn.predict(x_validation)
    accuracy = accuracy_score(y_validation, y_predict)
    score.append(accuracy)
    end = time.time()
    print("Complete time: " + str(end-start) + " Secs.")
print(score)
plt.plot(range(1,8), score)
plt.xlabel('value of k') 
plt.ylabel('accuracy')
plt.show
k = 1
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(orginal_x_train, orginal_y_train)
y_pred = knn.predict(orginal_test_data)
id =1353
print (y_pred[id])
plt.imshow(orginal_test_data.values[id,:].reshape((28, 28)))
plt.show()

print(len(y_pred))
result = pd.DataFrame({"ImageId" : list(range(1, len(y_pred)+1)), "Label" : y_pred})
result.head()
result.to_csv("Digit_Recogniser_Result",index=False, header=True)