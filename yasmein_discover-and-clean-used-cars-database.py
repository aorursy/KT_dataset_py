import pandas as pd

#load data

df=pd.read_csv('../input/autos.csv',encoding ="ISO-8859-1")

# view first 5 rows

df.head(5)
#show data types

df.dtypes
# describe data

df.describe()
df.seller.value_counts()