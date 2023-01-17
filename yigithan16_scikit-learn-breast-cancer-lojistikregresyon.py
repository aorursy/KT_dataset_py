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
from sklearn import preprocessing, datasets, neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Veriyi yükleyip bağımlı ve bağımsız değişkenleri belirledik
bcancer=datasets.load_breast_cancer()
X, y=bcancer.data[:,:2], bcancer.target

#Veriyi test ve eğitim kümesi olarak ikiye ayırmak için
X_egitim, X_test, y_egitim, y_test=train_test_split(X,y)
#Veriyi aynı ölçeğe indirmek için yapılan standartlaştırma işlemi
scaler=preprocessing.StandardScaler().fit(X_egitim)
X_egitim=scaler.transform(X_egitim)
X_test=scaler.transform(X_test)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix as cm
import seaborn as sns
import matplotlib.pyplot as plt
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_egitim, y_egitim)
predictions = classifier.predict(X_test)
score = round(accuracy_score(y_test, predictions), 3)
cm1 = cm(y_test, predictions)
sns.heatmap(cm1, annot=True, fmt=".0f")
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('Accuracy Score: {0}'.format(score), size = 15)
plt.show()

