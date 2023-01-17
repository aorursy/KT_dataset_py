import pandas as pd

import numpy as np

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.manifold import TSNE



import keras

from keras.models import Sequential

from keras.layers import Dense

from keras.models import load_model



import math



import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv('HR_comma_sep.csv')

len(df)