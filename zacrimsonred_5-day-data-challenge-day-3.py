

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

from scipy.stats import ttest_ind



df = pd.read_csv("../input/cereal.csv")

df_ttest = df[['type', 'sodium', 'potass']]

df_ttest.head(10)



Sodium_hot = df_ttest[df_ttest['type'] == "H"]['sodium']

Sodium_cold = df_ttest[df_ttest['type'] == "C"]['sodium']

print("Standard deviation for the sodium content of hot cereal is {:.3f} and std for the sodium content of cold cereal is {:.3f}".

     format(Sodium_hot.std(), Sodium_cold.std()))



ttest_ind(Sodium_hot, Sodium_cold, axis=0, equal_var=False)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14,8))

ax0, ax1 = axes.flatten()

ax0.hist(Sodium_hot, edgecolor = 'black')

ax0.set_title("Hot Cereal sodium content")

ax1.hist(Sodium_cold, edgecolor = 'black')

ax1.set_title("Cold Cereal sodium content")


