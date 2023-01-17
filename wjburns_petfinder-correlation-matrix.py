import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt
import seaborn as sb
%matplotlib inline
%%time
train = pd.read_csv('../input/train/train.csv')
test = pd.read_csv('../input/test/test.csv')
train.sample(5)
train.info()
test.info()
train.isnull().sum().sort_values(ascending = False)
test.isnull().sum().sort_values(ascending = False)
corr_matrix = train.corr()
corr_matrix['AdoptionSpeed'].sort_values(ascending = False)
plt.figure(figsize=(20,15))
sb.heatmap(corr_matrix)
plt.title("Correlation Matrix", size = 25)
correlation_cols = [
    'AdoptionSpeed',
    'Breed1',
    'Age',
    'Quantity',
    'Gender',
    'MaturitySize',
    'Health',
    'State',
    'VideoAmt',
    'Fee',
    'Color3',
    'Dewormed',
    'Breed2',
    'PhotoAmt',
    'Color2',
    'Color1',
    'Vaccinated',
    'Sterilized',
    'Type',
    'FurLength']

plt.figure(figsize=(15,10))
sb.heatmap(train[correlation_cols].corr(), annot = True);