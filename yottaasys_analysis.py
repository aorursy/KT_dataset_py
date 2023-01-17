# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from matplotlib import pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

path = '../input'

print(check_output(["ls", path]).decode("utf8"))





# Any results you write to the current directory are saved as output.
files = check_output(["ls", path]).decode('utf-8').splitlines()

for file_name in files:

    locals()[file_name[:-4]] = pd.read_csv(r'{0}/{1}'.format(path,file_name))
for df in files:

    print(df,locals()[df[:-4]].head(),end='\n',sep='\n')
products_with_aisle = pd.merge(products,aisles,on='aisle_id',how='right')

products_with_aisle.head()
products_grouped = products_with_aisle.groupby('aisle')['department_id'].count().reset_index()
plt.figure(figsize=(12,22))

sns.barplot(data=products_grouped,y='aisle',x='department_id')