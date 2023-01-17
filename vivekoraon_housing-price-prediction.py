import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
dataset = pd.read_csv('../input/housesalesprediction/kc_house_data.csv')
dataset.head()
dataset.info()