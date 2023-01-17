# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))





# Any results you write to the current directory are saved as output.



data = pd.read_csv("../input/train.tsv", sep = '\t')



#data.head()



data['item_condition_id'].value_counts()
# understanding relationship between item_condition_id and price

import matplotlib.pyplot as plt

plt.ylim(-50, 200)

import seaborn as sns

sns.violinplot(data=data, x = 'item_condition_id', y = 'price')

data.head()
# price distribution

plt.xlim(-50, 300)

sns.distplot(data['price'])
# mean price of popular brands

brands = data['brand_name'].value_counts()[:10].index

brands

k = data.groupby('brand_name').mean()['price']

k

p = k.loc[brands]

fig, ax = plt.subplots(figsize=(20, 10))

plt.ylabel('Mean price')

plt.bar(p.index, p)


