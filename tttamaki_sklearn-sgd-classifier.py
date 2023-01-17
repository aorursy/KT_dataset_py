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
#taken from

#Digit Recognizer in Python using Convolutional Neural Nets

#https://www.kaggle.com/kobakhit/digit-recognizer/digit-recognizer-in-python-using-cnn/comments



dataset = pd.read_csv("../input/train.csv")

target = dataset[[0]].values.ravel()

train = dataset.iloc[:,1:].values

test = pd.read_csv("../input/test.csv").values
from sklearn import linear_model, svm, metrics

classifier = linear_model.SGDClassifier(n_iter=100, n_jobs=6, penalty="l1")

print(classifier)
classifier.fit(train, target)
pred = classifier.predict(test)

np.savetxt('submission_of_mine.csv', np.c_[range(1,len(test)+1),pred], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')