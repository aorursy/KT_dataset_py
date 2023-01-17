# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np
%matplotlib notebook

import matplotlib.pyplot as plt
data = pd.read_csv("../input/train.csv")

data.head()
L = np.sqrt(784)

L
def plotNum(ind):

    plt.imshow(np.reshape(np.array(data.iloc[ind,1:]), (28, 28)), cmap="gray")
plt.figure()

for ii in range(1,17):

    plt.subplot(4,4,ii)

    plotNum(ii)
X = data.iloc[:, 1:]

y = data['label']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

from sklearn.linear_model import SGDClassifier



rfc = SGDClassifier(n_jobs=-1)

rfc.fit(X_train, y_train)
rfc.score(X_test, y_test)
unknown = pd.read_csv("../input/test.csv")
unknown.head()
y_out = rfc.predict(unknown)

y_out
Label = pd.Series(y_out,name = 'Label')

ImageId = pd.Series(range(1,28001),name = 'ImageId')

submission = pd.concat([ImageId,Label],axis = 1)

submission.to_csv('submission.csv',index = False)