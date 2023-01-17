# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

## WIRA DWI SUSANTO
## NIM: 17.01.53.0053

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

a = [1, 2, 3]
b = [4, 5, 6]
a + b
    
## BATAS

result = []
for first, second in zip(a, b):
    result.append(first + second)
result

## BATAS

a = np.array([1, 2, 3])
a

## BATAS

b = np.array([1, 2, 3.0])
b

## BATAS

type(a)

## BATAS

a = np.array([1, 2, 3])
a.dtype

## BATAS

a = np.array([1, 2, 3], dtype='int64')
a.dtype

## BATAS

a = np.array([1, 2, 3])
a.ndim

## BATAS

a = np.array([1, 2, 3])
a.shape

## BATAS

a = np.array([1, 2, 3])
f = np.array([1.1, 2.2, 3.3])
a + f

## BATAS

a * f

## BATAS

a ** f

## BATAS

np.sin(a)

## BATAS

a = np.array([1, 2, 3])
a[0]

## BATAS

a[0] = 10
a

## BATAS

a = np.array([1, 2, 3])
a.dtype

## BATAS

a[0] = 11.6
a

## BATAS

a = np.array([[0, 1, 2, 3], [10, 11, 12, 13]])
a

## BATAS

a.shape

## BATAS

a.size

## BATAS

a.ndim

## BATAS

a[1, 3]

## BATAS

a[1, 3] = -1
a

## BATAS

a = np.array([1, 2, 3])
a[1:2]

## BATAS

a[1:-1]

## BATAS

a[::2]

#END, LANJUT KE PANDAS

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
