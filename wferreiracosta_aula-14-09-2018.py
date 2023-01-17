import numpy as np # linear algebra
import pandas as pd # data processing CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
wine = pd.read_csv('../input/wine-reviews/winemag-data-130k-v2.csv')
iris = pd.read_csv('../input/iris-datasheet/Iris.csv')

wine.shape
wine.columns

wine['country'].value_counts()
wine.iloc[:,:]
wine.iloc[4:5,:]
