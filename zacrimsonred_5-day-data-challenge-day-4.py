import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt



df = pd.read_csv("../input/cereal.csv")

df.dtypes

dftest = df.head(20)

print(dftest)

name = dftest.name

calories = dftest.calories

y_pos = np.arange(len(name))

plt.bar(y_pos,calories)

plt.ylabel('Calories')

plt.title('Calory content on the first 20 cereals')
