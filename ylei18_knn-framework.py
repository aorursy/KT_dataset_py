import numpy as np # 数组常用库

import pandas as pd # 读入csv常用库

import matplotlib.pyplot as plt # 画图常用库
data_dir = '../input/'



def load_data(data_dir, train_rows):

    train = pd.read_csv(data_dir + 'train.csv')

    print(train.shape)

    X_train = train.values[0:train_rows, 1:]

    y_train = train.values[0:train_rows, 0]

    Pred_test = pd.read_csv(data_dir + 'test.csv')

    return X_train, y_train, Pred_test

    

train_row = 42000

Origin_X_train, Origin_y_train, Origin_X_test = load_data(data_dir, train_row)
#print(Origin_X_train.shape, Origin_y_train.shape, Origin_X_test.shape)

import matplotlib

import matplotlib.pyplot as plt

row = 3

print(Origin_y_train[row])

plt.imshow(Origin_X_train[row].reshape((28,28)))

plt.show()
classes = ["0","1","2","3","4","5","6","7","8",""]

from sklearn.model_selection import train_test_split

X_train, X_vali, y_train, y_vali = train_test_split(Origin_X_train, 

                                                    Origin_y_train, 

                                                    test_size = 0.2,

                                                   random_state = 1)

print(X_train.shape, X_vali.shape, y_train.shape, y_vali.shape)
import time

from sklearn.metrics import accuracy_score, classification_report,confusion_matrix 

from sklearn.neighbors import KNeighborsClassifier as KNN

ans_k = 0

k_range = range(1, 8)

scores = []

for k in k_range:

    print("k = " + str(k) +" begin ")

    start = time.time()

    model = KNN(n_neighbors = k)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_vali) #use validation set to predict

    accuracy = accuracy_score(y_vali, y_pred) #compare predicted result and validation result

    scores.append(accuracy)

    end = time.time()

    print(classification_report(y_vali, y_pred))

    print(confusion_matrix(y_vali, y_pred))

    

    print("Complete time: " + str(end-start) + " Secs.")
print(scores)

plt.plot(k_range, scores)

plt.xlabel('Value of K')

plt.ylabel('Testing accuracy')

plt.show()
k = 3 #pick best k from above

bestmodel = KNN(n_neighbors = k)

bestmodel.fit(Origin_X_train, Origin_y_train)

y_pred = bestmodel.predict(Origin_X_test)

# change this line to y_pred = knn.predict(Origin_X_test) for full test
print(y_pred[200])

plt.imshow(Origin_X_test[200].reshape((28,28)))

plt.show()
print(len(y_pred))



pd.DataFrame({"ImageId": list(range(1, len(y_pred) + 1)), "Label": y_pred}).to_csv('Digit_Recogniser_Result.csv',

                                                                                 index = False,

                                                                                 header = True)