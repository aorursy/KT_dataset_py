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
filepath = '../input/glass/glass.csv'
df = pd.read_csv(filepath)
df
target_col = 'Type'
print(df[target_col].unique())
print('-----------------------')
print(df[target_col].value_counts())
print('-----------------------')
print(df.dtypes)
print('-----------------------')
print(df.isna().sum())
y = df[target_col]
X = df.drop(columns=[target_col])
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

para = list(range(3, 10, 2))
print(para)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
results = {}
for n in para:
    print('para=', n)
    model = KNeighborsClassifier(n_neighbors=n)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    accu = accuracy_score(y_true=y_test, y_pred=preds)
    f1 = f1_score(y_true=y_test, y_pred=preds, average='micro')
    print(classification_report(y_true=y_test, y_pred=preds))
    print('--------------------------')
    results[n] = f1
import matplotlib.pylab as plt
# sorted by key, return a list of tuples
lists = sorted(results.items()) 
p, a = zip(*lists) # unpack a list of pairs into two tuples
plt.plot(p, a)
plt.show()
best_para = max(results, key=results.get)
print('best para', best_para)
print('value', results[best_para])