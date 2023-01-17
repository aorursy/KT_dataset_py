import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

df_train = pd.read_csv("../input/train.csv")

ncheckins_over_time = df_train.groupby("time").size()
plt.plot(ncheckins_over_time)
