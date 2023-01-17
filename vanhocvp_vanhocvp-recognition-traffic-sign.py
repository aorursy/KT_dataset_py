import numpy as np

import cv2 as cv

import glob



from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras.utils import np_utils

path_train = glob.glob('../input/proptit-aif-homework-1/final_train/final_train/*')

path_test  = glob.glob('../input/proptit-aif-homework-1/final_test/final_test/*')

numbers = [0, 2, 6, 10, 14, 22, 33, 34] # các label 

PATH = []

for i in numbers:

    path_tmp = "../input/proptit-aif-homework-1/final_train/final_train/" + str(i) + "/*"

    x = glob.glob(path_tmp)

    PATH.append(x)

#Đến đây PATH của chúng ta bây giờ sẽ là các path dẫn đến file ảnh train
l = 0

X = [] #data train

y = [] #label train

for k in PATH:

    a = [l]

    for i in k:

        x = cv.imread(i)

        x = cv.resize(x, (50, 50))

        n = np.asarray(x)

        X.append(n)

        y = np.concatenate((y, a), axis = 0) # tạo array label

    l = l+ 1

# X đang ở dạng list nên ta convert nó về array

X = np.asarray(X).reshape(-1, 50, 50, 3)

# one_hot_coding

y = np_utils.to_categorical(y, 8)
to_test = []

for img in path_test:

    x = cv.imread(img)

    x = cv.resize(x, (50, 50))

    n = np.asarray(x)

    to_test.append(n)

to_test = np.asarray(to_test).reshape(-1, 50, 50, 3)
def CNN(X_train,y_train, X_test):

    X_train = X_train/255.0 #giúp việc tính toán của chúng ta dễ dangf hơn vì các số trong khoảng [0, 1]

    X_test = X_test/255.0



    model = Sequential()



    model.add( Conv2D(filters = 32, kernel_size = (5, 5), activation = 'relu', padding = 'same') )

    model.add( Conv2D(filters = 32, kernel_size = (5, 5), activation = 'relu', padding = 'same') )

    model.add( MaxPooling2D(pool_size = (2, 2)))

    model.add( Dropout(0.25))



    model.add( Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu', padding = 'same') )

    model.add( Conv2D(filters = 32, kernel_size = (3, 3), activation = 'relu', padding = 'same') )

    model.add( MaxPooling2D(pool_size = (2, 2)))

    model.add( Dropout(0.25))



    model.add(Flatten())

    model.add(Dense(128, activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(64, activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(8, activation='softmax'))

    

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



    model.fit(X_train, y_train, batch_size=32,epochs = 5)



    label = model.predict(X_test)

    return label

y_pred = CNN(X, y,  to_test) # gọi hàm CNN



y_pred = np.asarray(y_pred).reshape((-1, 8))

label = np.argmax(y_pred, axis = 1)

res = []

for i in label:

    a = [numbers[i]]

    res = np.concatenate((res, a), axis = 0)

res  = np.asarray(res)

res = res.astype(int) #đưa về dạng số nguyên
import pandas as pd # để ghi file csv

import os # để đọc file

id = os.listdir('../input/proptit-aif-homework-1/final_test/final_test') # lấy tên các file ảnh, chuẩn bị cho output của bài toán

submission = pd.DataFrame({                                            

        "class": res,  "path": id

    })

submission.to_csv('vp.csv', index=False)
