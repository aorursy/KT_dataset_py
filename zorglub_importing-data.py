# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
filename = '../input/mnist_kaggle_some_rows.csv'
data = np.loadtxt(filename,
delimiter=',',
skiprows=2, 
usecols=[0,2],
dtype=str)

print(data) 
# Import package
import matplotlib.pyplot as plt
import numpy as np

# Assign filename to variable: file
file2 = '../input/digits/digits.csv'

# Load file as array: digits
digits = np.loadtxt(file2, delimiter=',')

# Print datatype of digits
print(type(digits))

# Select and reshape a row
im = digits[21, 1:]
print(im)
im_sq = np.reshape(im, (28, 28))
print(im_sq)

# Plot reshaped data (matplotlib.pyplot already loaded as plt)
plt.imshow(im_sq, cmap='Greys', interpolation='nearest')
plt.show()