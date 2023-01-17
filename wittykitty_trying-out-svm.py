# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
digit_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')



digit_df.head()
digit_df.info()
test_df.info()
import matplotlib.pyplot as plt
from sklearn import svm, decomposition, preprocessing
train_x = digit_df.values[:, 1:]

train_y = digit_df.ix[:,0]



print("size of training set: ", len(train_y))
from skimage.transform import rotate



ARTIFICIAL_DATA_SIZE = 2500

n = len(train_y)



rnd_indexes = np.random.randint(0, n, size=ARTIFICIAL_DATA_SIZE)

train_x = np.vstack((train_x, np.zeros((ARTIFICIAL_DATA_SIZE, train_x.shape[1]))))



k = n

for ind in rnd_indexes:

    img = train_x[ind, :].copy().reshape(28,28)

    angle = np.random.randint(-20, 20)

    img_rot = rotate(img, angle)

    train_x[k,:] = img_rot.reshape(784,)

    train_y[k] = train_y[ind]

    k = k + 1

print("size of training set after random rotations: ", len(train_y))

plt.title("last of added training examples, label: " + str(train_y[k-1]))

img1 = plt.imshow(train_x[k-1,:].copy().reshape(28,28), cmap=plt.cm.gray_r, interpolation="nearest")
COMPONENTS_RATIO= 0.8



train_x = preprocessing.scale(train_x)



pca = decomposition.PCA(n_components = COMPONENTS_RATIO, whiten = False)

train_x = pca.fit_transform(train_x)
from sklearn.cross_validation import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(train_x,train_y,random_state = 42,

                                                    test_size = 0.1)



classifier = svm.SVC(C=2.0)

classifier.fit(X_train, Y_train)

score = classifier.score(X_test, Y_test)

print("SVM score: ", score)