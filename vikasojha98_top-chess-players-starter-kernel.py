# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings as ws
ws.filterwarnings ("ignore")
sns.set_style("white")

df = pd.read_csv("/kaggle/input/world-top-chess-players-august-2020/top_chess_players_aug_2020.csv")
df.head()
df.info()
top_ten_players = df[:10]
top_ten_players
title_dist = df.Title.value_counts().reset_index()
print(title_dist)
plt.figure(figsize=(8,5))
plt.title("Title distribution of Players")
sns.barplot(x = "index", y = "Title", data = title_dist)
plt.show()