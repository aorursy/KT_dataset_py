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
constant = 8*(3.14159)/(((6.626*10**(-34))**3)*((3*10**8)**2))
print(constant)
J = 1.60218e-19  #eV
z = []
x = 0
for x in range(0,1000,1):
    x += 0.01
    f =  constant*(((x*J)**3)/((2.7**((x*J)/((1.38064*10**(-23))*(5778))))-1))
    z.append(f)
    print(f)
d = (1*J)**3
print(d)
e = ((2.7**((1*J)/((1.38064*10**(-23))*(5778))))-1)
print(e)
f = d/e
print(f)
