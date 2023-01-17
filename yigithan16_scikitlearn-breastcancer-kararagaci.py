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
from sklearn import tree
from sklearn import metrics
from sklearn.metrics import confusion_matrix as cm
import seaborn as sns
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn import preprocessing, datasets, neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
bcancer=datasets.load_breast_cancer()
X=bcancer.data
y=bcancer.target

print('Class labels:', np.unique(y))
#Veriyi normalize ediyoruz
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
ss.fit(X)
X=ss.transform(X)
#Veri setimizi train ve test diye ikiye ayırıyoruz 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
#Karar ağacı algoritması ve elde edilen sonuçlar
dtree=tree.DecisionTreeClassifier(criterion='entropy', random_state=0)
dtree.fit(X_train, y_train)
predictions = dtree.predict(X_test)
score = round(accuracy_score(y_test, predictions), 3)
cm1 = cm(y_test, predictions)
sns.heatmap(cm1, annot=True, fmt=".0f")
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('Accuracy Score: {0}'.format(score), size = 15)
plt.show()
#Karar ağacının oluşturulması 
import graphviz
dot_data = tree.export_graphviz(dtree, out_file=None, feature_names=bcancer.feature_names,
                               class_names=bcancer.target_names, filled=True, rounded=True,
                               special_characters=True)
graph=graphviz.Source(dot_data)


dot_data = tree.export_graphviz(dtree, out_file=None,
                               feature_names=bcancer.feature_names,
                               class_names=bcancer.target_names,
                               filled=True, rounded=True,
                               special_characters=True)
graph=graphviz.Source(dot_data)
graph