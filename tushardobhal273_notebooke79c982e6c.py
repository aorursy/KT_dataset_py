# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

from sklearn.preprocessing import LabelBinarizer

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split

import os

os.listdir('../input/ghouls-goblins-and-ghosts-boo')

train= pd.read_csv('../input/ghouls-goblins-and-ghosts-boo/train.csv.zip', compression='zip')

test= pd.read_csv('../input/ghouls-goblins-and-ghosts-boo/test.csv.zip', compression='zip')

train.head(10)
test.head(10)
trainid=train.values[:,0]

testid=test.values[:,0]

y= train['type']

x = pd.get_dummies(train.drop(['color','type','id'], axis = 1))

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

X=np.array(X_train)

y=np.array(y_train)
alg=MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,), random_state=0).fit(X,y)

print(alg)

y_pred=alg.predict(X_test)
y_pred=alg.predict(X_test)

alg.score(X_test,y_test)
from sklearn.model_selection import cross_val_score

scores= cross_val_score(alg, X, y, cv=5)

scores

scores.mean()
test1=test.drop(['color','id'],axis=1)

prediction=alg.predict(test1)



submission=pd.DataFrame({'id':testid, 'type': prediction})

submission.to_csv("submission.csv",index=False)

submission.head()