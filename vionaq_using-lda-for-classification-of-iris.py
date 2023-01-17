import pandas as pd

from sklearn.preprocessing import LabelEncoder

from matplotlib import pyplot as plt

import numpy as np

import math
df = pd.io.parsers.read_csv("../input/Iris.csv", sep=',')

print (df)
feature_dict = {i:label for i, label in zip(range(5), ('Id','sepal length in cm', 'sepal width in cm', 'petal length in cm', 'petal width in cm'))}

df.columns = [l for i,l in sorted(feature_dict.items())] + ['class label']

df.dropna(how='all', inplace=True) #drop the extra line in the csv file