# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_1 = pd.read_csv('/kaggle/input/summer-products-and-sales-in-ecommerce-wish/summer-products-with-rating-and-performance_2020-08.csv')
df_1.head(5)
df_1.info()
df_1=df_1.drop(columns=['merchant_profile_picture','merchant_has_profile_picture','merchant_id','title','title_orig',

                        'merchant_id','merchant_title','merchant_name','product_id','product_url','product_picture',

                       'urgency_text','tags','product_color','product_variation_size_id','shipping_option_name',

                        'origin_country','merchant_profile_picture','crawl_month','has_urgency_banner','currency_buyer','theme'])
df_1.head(5)
df_1.isnull().sum()
df_1.drop_duplicates(inplace=True)
df_1
plt.figure(figsize=(15,15))

corrmat = df_1.corr()

corrmat = np.tril(corrmat)

corrmat[corrmat==0] = None

corrmat = corrmat.round(1)

labels = df_1.select_dtypes(include='number').columns.values

f, ax = plt.subplots(figsize=(15, 8))

sns.heatmap(corrmat, annot=True, vmax=0.8,vmin=-0.8, cmap='seismic_r', xticklabels=labels,yticklabels=labels, cbar=False)

plt.legend('')



plt.show()
df_1.hist(bins=40, figsize=(20,15))

plt.show()