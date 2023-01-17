# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

import pandas as pd
import numpy as np
import warnings as ws
ws.filterwarnings ("ignore")
df = pd.read_csv("/kaggle/input/indian-chess-grandmasters/indian_grandmasters_july_2020.csv")
df.head()
df.info()
print("Top 10 Indian Chess Grandmasters")
top_ten_players = df.sort_values(by='Classical_Rating', ascending=False)[:10].reset_index(drop=True)
top_ten_players
