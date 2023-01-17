import numpy as np # 数组常用库

import pandas as pd # 读入csv常用库

import matplotlib.pyplot as plt # 画图常用库
data_dir = "../input/"



def load_data(data_dir, train_row):

    train = pd.read_csv(data_dir + "train.csv")

    data = train.values[0:train_row,1:] # 取下标为1-784的列（pixel0 - pixel783）

    label = train.values[0:train_row,0] # 取下标为0的列 (label)

    test_data = pd.read_csv(data_dir + "test.csv").values  

    return data, label, test_data



train_row = 1213 

data, label, test_data = load_data(data_dir, train_row)
import matplotlib

import matplotlib.pyplot as plt



classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

rows = 6



for y, e in enumerate(classes):

    idxs = np.nonzero([i == y for i in label])

    idxs = np.random.choice(idxs[0], rows)

    for i , idx in enumerate(idxs):

        plt_idx = i * len(classes) + y + 1

        plt.subplot(rows, len(classes), plt_idx)

        plt.imshow(data[idx].reshape((28, 28)))

        plt.axis("off")

        if i == 0:

            plt.title(e)

plt.show()
from sklearn.model_selection import train_test_split



X_train,X_vali, y_train, y_vali = train_test_split(data,

                                                   label,

                                                   test_size = 0.23,

                                                   random_state = 13)



print(X_train.shape, X_vali.shape, y_train.shape, y_vali.shape)

import time

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

from sklearn.neighbors import KNeighborsClassifier

k_range = range(1, 8)

scores = []



for k in k_range:

    print("k = " + str(k) + " begin ")

    start = time.time()

    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(X_train,y_train)

    y_pred = knn.predict(X_vali)

    accuracy = accuracy_score(y_vali,y_pred)

    scores.append(accuracy)

    end = time.time()

    print(classification_report(y_vali, y_pred))  

    print(confusion_matrix(y_vali, y_pred))  

    

    print("Complete time: " + str(end-start) + " Secs.")

print (scores)

plt.plot(k_range,scores)

plt.xlabel('Value of K')

plt.ylabel('Testing accuracy')

plt.show()
k = 4



knn = KNeighborsClassifier(n_neighbors=k)

knn.fit(data,label)

y_pred = knn.predict(data)
print (y_pred[200])

plt.imshow(data[200].reshape((28, 28)))

plt.show()
print(len(y_pred))



# save submission to csv

pd.DataFrame(

    {"ImageId": list(range(1,len(y_pred)+1)),"Label": y_pred}

).to_csv('Digit_Recogniser_Result.csv', index=False,header=True)
