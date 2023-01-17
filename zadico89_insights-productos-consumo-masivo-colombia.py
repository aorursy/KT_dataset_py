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
df = pd.read_excel("/kaggle/input/productos-consumo-masivo/output - Kaggle.xlsx")

df.head()
df.shape
df.dtypes
for col in df:

    print(">>Columna: {}".format(col))

    print(df[col].value_counts(dropna=False))

    print("______________")
columnas_interes = ["date", "prod_name", "prod_name_long", "prod_brand", "subcategory", "tags", "prod_unit_price", "prod_units", "prod_source"]

df = df[columnas_interes]

df.head()
# df.prod_unit_price.astype(float)

# Error: could not convert string to float: '70386,95' df["prod_unit_price"] = 

df[df["prod_unit_price"] == "70386,95"] # index=15821
df["prod_unit_price_float"] = df["prod_unit_price"].apply(lambda x: float(str(x).replace(',', '.')))
df.loc[15821,"prod_unit_price_float"]
type(df.loc[15821,"prod_unit_price_float"])
print("Mean: {}".format(df.prod_unit_price_float.mean()))

print("Var: {}".format(df.prod_unit_price_float.var()))

print("Min: {}".format(df.prod_unit_price_float.min()))

print("Max: {}".format(df.prod_unit_price_float.max()))
df.prod_unit_price_float.plot.box()
df['prod_unit_price_float'].plot.hist()
df['prod_unit_price_float'].plot.kde()
outliers = df[df['prod_unit_price_float'] >= 3*df['prod_unit_price_float'].std()]

outliers.shape
outliers = outliers.sort_values(by='prod_unit_price_float', ascending=False)

outliers
outliers.head(50)
prods_err_prec = outliers[outliers['prod_unit_price_float'] > 499999]

prods_err_prec.shape