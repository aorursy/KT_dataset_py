import numpy as np

import pandas as pd



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline  
def check_date(spacing,iteration):

    for i in range(1,iteration+1):

        train_df = pd.read_csv('../input/train.csv', skiprows=range(1, spacing*i),nrows=1)

        print('#'+str(i),train_df['fecha_dato'])

        

spacing = 13000000

iteration = 1

check_date(spacing,iteration)

print(spacing*iteration)

print('we reach the May of 2016 @ about 13,000,000th row.')
hist = pd.DataFrame()

rang = [1000000*i for i in range(10,14)]



for i in rang:

    train_df = pd.read_csv('../input/train.csv', skiprows=range(1, i),nrows=500000)

    string = 'mean' + ' ' + str(i)

    hist[string] = train_df.describe().transpose()['mean'][11:35] #.plot(kind='bar',legend=True)

    

hist.plot.bar(figsize=(10,10))

#train_df.info()