from sklearn import datasets

import numpy as np



iris = datasets.load_iris()

X = iris.data[:, [2, 3]]

y = iris.target



print('Class labels:', np.unique(y))
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(

         X, y, test_size=0.3, random_state=0)
from sklearn.preprocessing import StandardScaler



sc = StandardScaler()

sc.fit(X_train)

X_train_std = sc.transform(X_train)

X_test_std = sc.transform(X_test)
from IPython.display import Image

Image(filename='../input/03_07.png', width=700) 
Image(filename='../input/03_08.png', width=600) 
import matplotlib.pyplot as plt

import numpy as np



np.random.seed(0)

X_xor = np.random.randn(200, 2)

y_xor = np.logical_xor(X_xor[:, 0] > 0,

                       X_xor[:, 1] > 0)

y_xor = np.where(y_xor, 1, -1)



plt.scatter(X_xor[y_xor == 1, 0],

            X_xor[y_xor == 1, 1],

            c='b', marker='x',

            label='1')

plt.scatter(X_xor[y_xor == -1, 0],

            X_xor[y_xor == -1, 1],

            c='r',

            marker='s',

            label='-1')



plt.xlim([-3, 3])

plt.ylim([-3, 3])

plt.legend(loc='best')

plt.tight_layout()

# plt.savefig('./figures/xor.png', dpi=300)

plt.show()
Image(filename='../input/03_11.png', width=700) 