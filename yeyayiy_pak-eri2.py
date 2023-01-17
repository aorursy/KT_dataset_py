# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
a = [1, 2, 3]
b = [4, 5, 6]
a + b

result = []
for first, second in zip(a, b):
    result.append( first + second)
    result
import numpy as np
a = np.array ([1,2,3])
a
b = np.array([1, 2, 3.0])
b
type(a)
a = np.array([1, 2, 3])
a.dtype
a = np.array([1, 2, 3], dtype='int64')
a.dtype
a = np.array ([1, 2, 3])
a.ndim
a = np.array([1, 2, 3])
a.shape
a = np.array([1, 2, 3])
f = np.array([1.1, 2.2, 3.3])
a+f
a * f
a ** f
np.sin(a)
a = np.array([1, 2, 3])
a[0]
a[0] = 10
a
a = np.array([1, 2, 3])
a.dtype
a[0] = 11.6
a

a = np.array([[ 0, 1, 2, 3],
            [10, 11, 12, 13]])
a
a.shape
a.size
a.ndim
a[1, 3]
a[1, 3]= -1
a
a = np.array([1, 2, 3])
a[1:2]
a[1:-1]
a[::2]
import pandas as pd
obj = pd.Series([1, 2, 3])
obj
type(obj)
obj2 = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
obj2
obj2.index
obj2.values
obj2['a']
obj2['a']=4
obj2
obj2[['a', 'c']]
obj3 = pd.Series([4, 5, 6], index= ['a', 'b', 'c'])
obj3
obj2 + obj3
data = {'kota': ['semarang','semarang','semarang','bandung','bandung','bandung'],'tahun': [2016, 2017, 2018, 2016, 2017, 2018],'populasi':[1.5, 2.1, 3.2, 2.3, 3.2, 4.5]}
frame = pd.DataFrame(data)
frame
type(frame)
frame.shape
frame.info()
frame.head()
frame.tail()
frame.columns
frame.index
frame.values
frame.describe()
frame['kota'].value_counts()
frame['populasi']
frame.loc[2]
frame.loc[2:3]
frame.loc[2]
frame['populasi'][2]
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np

x = np.linspace(0,2*np.pi, 100)
cos_x = np.cos(x)
fig, ax = plt.subplots()
_ = ax.plot(x,cos_x)
fig, ax = plt.subplots()
_ = ax.plot(x,cos_x)
_ = ax.set_aspect ('equal')
fig, ax = plt.subplots()
_=ax.plot(x, cos_x, markersize=20, linestyle='-.',color='red',label='cos')
_=ax.set_aspect('equal')
_=ax.legend()
x = np.array([1,2,3])
y = np.array([4,5,6])

fig, ax = plt.subplots()
_=ax.scatter(x,y)
_=ax.set_xlabel('x axis')
_=ax.set_ylabel('y axis')
kategori = ['panas','dingin']
jumlah = [8,5]

fig, ax = plt.subplots()
_ = ax.bar(kategori, jumlah)
_= ax.set_xlabel('Kategori')
_= ax.set_ylabel('Jumlah')
_= ax.set_title('Penikmat kopi')