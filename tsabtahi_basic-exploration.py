# numpy, pandas

import numpy as np 

import pandas as pd 

import datetime

import numpy as np



# plots

import matplotlib.pyplot as plt

%matplotlib inline

import matplotlib

matplotlib.style.use('ggplot')





from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

df_ac= pd.read_csv('../input/database.csv')

# Supress unnecessary warnings so that presentation looks clean

import warnings

warnings.filterwarnings('ignore')

#Print all rows and columns. Dont hide any

pd.set_option('display.max_rows', None)

pd.set_option('display.max_columns', None)



import sklearn.utils

df= sklearn.utils.shuffle(df_ac)



df.head(10)
df['Year'].value_counts().sort_index(ascending=True).plot(kind='bar')