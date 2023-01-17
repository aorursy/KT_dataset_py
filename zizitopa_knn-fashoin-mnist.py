# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
path = "/kaggle/input/jds101/"
data = pd.read_csv(path + 'fashion-mnist_train.csv')

test = pd.read_csv(path + 'new_test.csv')

data.head()
data.groupby('label').count()
test.head()
def print_img(line):

    img = np.array(line).reshape(28, -1)

    plt.imshow(img, cmap="gray")
line = data.iloc[4]

img = np.array(line[1:])

print_img(img)
from sklearn.neighbors import KNeighborsClassifier
X_knn_train = np.array(data.iloc[:, 1:])

y_knn_train = np.array(data.iloc[:, 0])

X_knn_train.shape, y_knn_train.shape
from sklearn.decomposition import PCA
pca = PCA()

X_pca = pca.fit(X_knn_train)
X_t = X_pca.transform(X_knn_train)

X_t.shape
knn = KNeighborsClassifier(weights='distance' , n_neighbors=11)

knn.fit(X_t, y_knn_train)
X_knn_test = np.array(test)

X_knn_test.shape
X_test = X_pca.transform(X_knn_test)
answer_knn = knn.predict(X_test)

answer_knn.shape
print_img(X_knn_test[2])
answer_knn[2]
answers = answer_knn.reshape(10000, 1)
b = np.arange(1, 10001).reshape(10000, 1)

b
result = np.concatenate((b, answers), axis=1)
df = pd.DataFrame(result)
df.head()
df.columns = ['id', 'label']
df.to_csv("answers.csv", index=False)
knn.score(X_knn_train, y_knn_train)