# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# to read a parquet file

file = pd.read_parquet("/kaggle/input/msr-2020-3k/person.parquet")

#Print the file

file
#Execute basic commands

# To know the number of rows and columns

#file.shape 

file.shape
#To get id greater than 100000

res = file[file['id']>100000]

res
file.head()
temp = pd.read_parquet("/kaggle/input/msr-2020-3k/directory_entry_dir.parquet")

temp