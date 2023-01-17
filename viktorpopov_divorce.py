# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('/kaggle/input/divorce-predictors-data-set-csv/divorce-csv.csv', sep=',')
df.head()
df.shape
sns.countplot('Class', data=df)
plt.ylabel('распределение классов')
plt.show()
#Create a correlation matrix
corr = df.corr().abs()
#Select the upper triangle of the correlation matrix
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
upper.head(10)
#Find the index of feature columns with a correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
to_drop
hug_cor_data = df.drop(['Atr19', 'Atr20', 'Atr36'], axis=1)
hug_cor_data.shape
X = hug_cor_data.iloc[:, :-1]
y = hug_cor_data.Class
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 11)
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
