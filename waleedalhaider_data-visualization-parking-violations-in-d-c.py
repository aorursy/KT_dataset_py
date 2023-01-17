import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

%matplotlib inline
df = pd.read_csv('../input/parking.csv')

df.head(n=3)
a = df['RP_PLATE_STATE'].value_counts()

print (a)
b = df['VIOLATION_DESCRIPTION'].value_counts()

print (b)
c = df['DAY_OF_WEEK'].value_counts()

print (c)
d = df['BODY_STYLE'].value_counts()

print (d)
e = df['LOCATION'].value_counts()

print (e)
f = df['OBJECTID'].value_counts()

print (f)
g = df['ADDRESS_ID'].value_counts()

print (g)
df.plot(x='ADDRESS_ID', y='OBJECTID', style='.')
threedee = plt.figure().gca(projection='3d')

threedee.scatter(df.index, df['ADDRESS_ID'], df['OBJECTID'])

threedee.set_xlabel('Index')

threedee.set_ylabel('ADDRESS ID')

threedee.set_zlabel('OBJECTID')

plt.show()