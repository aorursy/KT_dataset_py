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
df = pd.read_csv("/kaggle/input/mushroom-classification/mushrooms.csv")

print("Rows: ", df.shape[0])
print("Columns: ", df.shape[1],"\n","\n" )
print(df.head())

from sklearn import preprocessing

le=preprocessing.LabelEncoder()

for column in df.columns:
    df[column] = le.fit_transform(df[column])
        
print(df.head())


X = df[['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
       'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
       'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
       'stalk-surface-below-ring', 'stalk-color-above-ring',
       'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
       'ring-type', 'spore-print-color', 'population', 'habitat']]
Y = df["class"]

print("conjunto de individuos sin etiquetar:","\n","\n",X,"\n","\n","\n","\n")
print("conjunto de etiquetas de clase:","\n","\n",Y,"\n")
 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=1/3, random_state=0)

print("x_train:","\n","\n",x_train,"\n","\n","\n","\n")
print("x_test:","\n","\n",x_test,"\n","\n","\n","\n")
print("y_train:","\n","\n",y_train,"\n","\n","\n","\n")
print("y_test:","\n","\n",y_test,"\n")
from sklearn.ensemble import RandomForestClassifier

clf_rf=RandomForestClassifier(n_estimators= 10,max_features = 'sqrt',max_samples = 2/3, random_state=0)

clf_rf.fit(x_train, y_train)

y_pred=clf_rf.predict(x_test)

print("y_pred:","\n","\n",y_test,"\n")
from sklearn.metrics import classification_report, confusion_matrix

print("Matriz de confusión:","\n ","\n ",confusion_matrix(y_test,y_pred),"\n ","\n ","\n ","\n ")

print("Rendimiento",":\n", classification_report(y_test,y_pred), "\n " )
from sklearn.neural_network import MLPClassifier

clf_rn= MLPClassifier(hidden_layer_sizes=(100),activation="relu",solver="lbfgs", max_iter=200, random_state=0) 

clf_rn.fit(x_train, y_train)

y_pred_rn=clf_rn.predict(x_test)

print("y_pred_rn:","\n","\n",y_test,"\n")
from sklearn.metrics import mean_squared_error

print("Matriz de confusión","\n","\n",confusion_matrix(y_test,y_pred_rn),"\n ","\n ","\n ","\n " )
print("Indicadores","\n","\n", classification_report(y_test,y_pred_rn),"\n ","\n ","\n ","\n " )
print("Error cuadrático medio:","\n","\n",mean_squared_error(y_test, y_pred_rn),"\n" )