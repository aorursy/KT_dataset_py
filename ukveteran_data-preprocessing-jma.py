import pandas as pd

data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data', header=None)

data.columns = ['Sample code', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',

                'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',

                'Normal Nucleoli', 'Mitoses','Class']



data = data.drop(['Sample code'],axis=1)

print('Number of instances = %d' % (data.shape[0]))

print('Number of attributes = %d' % (data.shape[1]))

data.head()
import numpy as np



data = data.replace('?',np.NaN)



print('Number of instances = %d' % (data.shape[0]))

print('Number of attributes = %d' % (data.shape[1]))



print('Number of missing values:')

for col in data.columns:

    print('\t%s: %d' % (col,data[col].isna().sum()))
data2 = data['Bare Nuclei']



print('Before replacing missing values:')

print(data2[20:25])

data2 = data2.fillna(data2.median())



print('\nAfter replacing missing values:')

print(data2[20:25])
print('Number of rows in original data = %d' % (data.shape[0]))



data2 = data.dropna()

print('Number of rows after discarding missing values = %d' % (data2.shape[0]))
%matplotlib inline



data2 = data.drop(['Class'],axis=1)

data2['Bare Nuclei'] = pd.to_numeric(data2['Bare Nuclei'])

data2.boxplot(figsize=(20,3))
Z = (data2-data2.mean())/data2.std()

Z[20:25]
print('Number of rows before discarding outliers = %d' % (Z.shape[0]))



Z2 = Z.loc[((Z > -3).sum(axis=1)==9) & ((Z <= 3).sum(axis=1)==9),:]

print('Number of rows after discarding missing values = %d' % (Z2.shape[0]))
dups = data.duplicated()

print('Number of duplicate rows = %d' % (dups.sum()))

data.loc[[11,28]]
print('Number of rows before discarding duplicates = %d' % (data.shape[0]))

data2 = data.drop_duplicates()

print('Number of rows after discarding duplicates = %d' % (data2.shape[0]))