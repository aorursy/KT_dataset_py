# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os


# Any results you write to the current directory are saved as output.
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
print(os.listdir("../input"))
data = pd.read_csv("../input/8V280L8VQ-clash-royale-da.csv")
data.head()
trimmed_df = data[["my_result","my_trophies","opponent_trophies","my_deck_elixir","op_deck_elixir","my_troops","my_buildings","my_spells","op_troops","op_buildings","op_spells","my_commons","my_rares","my_epics","my_legendaries","op_commons","op_rares","op_epics","op_legendaries"]]
trimmed_df.head()
class_col = trimmed_df[["my_result"]]
class_col.head()
my_label_encoder = preprocessing.LabelEncoder()
encoded_class_array = my_label_encoder.fit_transform(class_col.iloc[:,0])
without_class_df = trimmed_df.drop("my_result",axis=1)
dataArray = without_class_df.values
TrainData,TestData,TrainClass,TestClass = train_test_split(dataArray,encoded_class_array,test_size=0.30)
knnModel = KNeighborsClassifier(n_neighbors=5)
knnModel.fit(TrainData, TrainClass)
predictedClassFromTestData = knnModel.predict(TestData)
print("Accuracy: ",metrics.accuracy_score(TestClass, predictedClassFromTestData))
mytest = knnModel.predict(np.array([[2513,2477,3.625,3.875,5,1,2,7,0,1,4,1,2,1,1,2,4,1]]))
print(mytest)
#2=Win
#0=Loss
#1=Draw
