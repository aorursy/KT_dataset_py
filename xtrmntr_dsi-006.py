from sklearn import preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
url = "https://raw.githubusercontent.com/edlich/eternalrepo/master/DS-WAHLFACH/dsm-beuth-edl-demodata-dirty.csv"
dataFrame = pd.read_csv(url)
dataFrame
df2 = dataFrame.drop("full_name", axis=1)
df2
df3 = df2.drop("id", axis=1)
df3
df3.isnull()
df3.dropna()
df3.duplicated()
#Remove duplicates
df3.drop_duplicates(keep=False, inplace=True)
df3.duplicated()
df3
df3.isnull().any()
df3[df3["email"].isnull()]
df3[df3["gender"].isnull()]
#Replace the missing gender by female
df3.loc[df3["gender"].isnull(), "gender"] = "Female"

#remove the other dataset in email"
df3 = df3[pd.notnull(df3["email"])]

df3
### first of all - check dataType
df3.dtypes["age"]
## getting the entry that contains incorrect data
theRow = df3[pd.to_numeric(df3["age"], errors="coerce").isnull()]
theRow
#replacing it
df3["age"] = df3["age"].replace("old", 70)
df3
df3["age"] = pd.to_numeric(df3["age"])
df3.dtypes["age"]
num = df3["age"]._get_numeric_data()
num[num < 0] = num * -1
df3
df3["gender"] = df3["gender"].map({"Female":"F", "Male":"M"})
df3["gender"]
dataFrame
df3