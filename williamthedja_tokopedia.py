# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib as mpl



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/tokopedia-product-reviews/tokopedia_product_reviews/product_reviews_dirty.csv')

df.head()
df.info()
jenis = df.groupby(['category'])['product_id'].count()

jenis
ambildata = df.iloc[:20]

ambildata['product_name']

 
ambildata.rating
jenis = df.groupby(['category'])['product_id'].count()



plt.bar(jenis.index, jenis)

plt.ylabel("Jumlah masing - masing produk")

plt.xlabel("Jenis produk yang ada")

plt.title("Jumlah produk yang ada berdasarkan kategori")

plt.show()
jumlahprodukterjual = df.groupby(['product_name'])['sold'].count().iloc[:20]

#jumlahprodukterjual.sort_values()

#jumlahprodukterjual

plt.barh(jumlahprodukterjual.index, jumlahprodukterjual)

plt.ylabel("nama produk")

plt.xlabel("Jumlah")

plt.title("Jumlah produk terjual pada 20 data pertama")

plt.show()
jumlahprodukterjualberdasarkanrating = df.groupby(['rating'])['sold'].count()

plt.bar(jumlahprodukterjualberdasarkanrating.index, jumlahprodukterjualberdasarkanrating)

plt.ylabel("jumlah")

plt.xlabel("rating")

plt.title("Jumlah produk terjual berdasarkan rating")

plt.show()