# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np



import matplotlib.pyplot as plt



%matplotlib inline
df = pd.read_csv('../input/voice.csv')

df.head()
df.corr()
df.isnull().sum()
df.shape[0]
print("Total number of labels: {}".format(df.shape[0]))

print("Number of male: {}".format(df[df.label == 'male'].shape[0]))

print("Number of female: {}".format(df[df.label == 'female'].shape[0]))
df.shape
colors=('r','b')

df.plot(kind='scatter', x='meanfreq', y='sd',c=colors)
X=df.iloc[:, :-1]

X.head()
from sklearn.preprocessing import LabelEncoder

y=df.iloc[:,-1]



# Encode label category

# male -> 1

# female -> 0



gender_encoder = LabelEncoder()

y = gender_encoder.fit_transform(y)

y
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
from sklearn.preprocessing import StandardScaler

StandardScaler().fit_transform(X_train,y_train)

StandardScaler().fit_transform(X_test,y_test)

np.mean(X_train)

np.std(X_train)

np.std(X_test)

np.mean(X_test)
from sklearn.svm import SVC

for k in range(31):

    svc = SVC(kernel='linear', C=k).fit(X_train,y_train)

    print('Test Accuracy: %.3f' % svc.score(X_test, y_test))

         
C_hyperparameter()
from sklearn.model_selection import cross_val_score



scores = cross_val_score(estimator=SVC(kernel='linear',C=10),

                        X=X_train,

                        y=y_train,

                        cv=10,

                        n_jobs=1)



print('Cross validation scores: %s' % scores)



import matplotlib.pyplot as plt

plt.title('Cross validation scores')

plt.scatter(np.arange(len(scores)), scores)

plt.axhline(y=np.mean(scores), color='r') # Mean value of cross validation scores