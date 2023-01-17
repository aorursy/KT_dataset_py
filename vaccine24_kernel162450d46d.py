import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



df = pd.read_csv (r'../input/corona19-march-192020/full_data.csv')



print(df.groupby('new_cases').size())







print(df.describe())




