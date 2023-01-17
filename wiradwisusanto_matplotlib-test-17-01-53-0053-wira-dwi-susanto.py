# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

## WIRA DWI SUSANTO
## NIM: 17.01.53.0053

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

x = np.linspace(0,2 * np.pi, 100)
cos_x = np.cos(x)

fig, ax = plt.subplots()
_ = ax.plot(x, cos_x)

## BATAS

fig, ax = plt.subplots()
_ = ax.plot(x, cos_x)
_ = ax.set_aspect('equal')

## BATAS

fig, ax = plt.subplots()
_ = ax.plot(x, cos_x, markersize=20, linestyle='-.',
           color='red', label='cos')
_ = ax.set_aspect('equal')
_ = ax.legend()

## BATAS

x = np.array([1, 2, 3])
y = np.array([4, 5, 6])

fig, ax = plt.subplots()
_ = ax.scatter(x, y)
_ = ax.set_xlabel('x axis')
_ = ax.set_ylabel('y axis')

## BATAS

kategori = ['Panas', 'Dingin']
jumlah = [8, 5]

fig, ax = plt.subplots()
_ = ax.bar(kategori, jumlah)
_ = ax.set_xlabel('Lategori')
_ = ax.set_ylabel('Jumlah')
_ = ax.set_title('Penikmat Kopi')
        
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
