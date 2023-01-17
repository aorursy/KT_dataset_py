import IPython.display as display

from PIL import Image

import numpy as np

import matplotlib.pyplot as plt

import os

import cv2
CATEGORIES = [str(i) for i in range(10)]

PATH = "/kaggle/input/chars74digits/Fnt"
traning_data = []

for category in CATEGORIES:

    path = os.path.join(PATH, category)

    class_num = CATEGORIES.index(category)

    for img in os.listdir(path):

        image_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)

        resized_array = cv2.resize(image_array, (64, 64))

        _, resized_array = cv2.threshold(resized_array, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        traning_data.append([resized_array, class_num])

len(traning_data)
import numpy as np

import matplotlib.pyplot as plt



fig = plt.figure(figsize=(9, 8))

rows, columns = 5, 10

ax = []

for i in range(columns*rows):

    # create subplot and append to ax

    ax.append(fig.add_subplot(rows, columns, i+1) )

    ax[-1].set_title("ax:"+str(i))  # set title

    plt.imshow(traning_data[6000+i][0], cmap='gray')

    plt.axis("off")



plt.show()  # finally, render the plot
X = []

y = []

for features, label in traning_data:

    X.append(features)

    y.append(label)



X = np.array(X).reshape(-1,4096)

X[0]
from sklearn import model_selection

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.20)

print(len(X_train), len(X_test))
plt.imshow(X_train[5].reshape(-1,64))

print(y_train[5])
from sklearn import svm

from sklearn import metrics

clf = svm.SVC()

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = metrics.accuracy_score(y_test, y_pred)

acc