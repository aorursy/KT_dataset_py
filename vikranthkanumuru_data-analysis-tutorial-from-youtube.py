!git clone https://github.com/KeithGalli/Pandas-Data-Science-Tasks.git
import pandas as pd

import matplotlib.pyplot as plt

import glob
path = r'/kaggle/working/Pandas-Data-Science-Tasks/SalesAnalysis/Sales_Data' # use your path

all_files = glob.glob(path + "/*.csv")



li = []



for filename in all_files:

    df = pd.read_csv(filename, index_col=None, header=0)

    li.append(df)



all_months_data = pd.concat(li, axis=0, ignore_index=True)
all_data = all_months_data

all_data
all_data=all_data.dropna(how="all")

nan_df=all_data[all_data.isna().any(axis=1)]

nan_df.head()
all_data["Quantity Ordered"]= pd.to_numeric(all_data["Quantity Ordered"])

all_data["Price Each"]=pd.to_numeric(all_data["Price Each"])
all_data=all_data[all_data["Order Date"].str[0:2]!="Or"]
all_data["Month"]=all_data["Order Date"].str[0:2]

all_data.head()
all_data["month"]=all_data["Month"].astype("int32")

all_data["Month"].head()

all_data["sales"]= all_data["Quantity Ordered"]*all_data["Price Each"]

all_data["sales"].head()
all_data.groupby('Month').sum()
all_data
all_data["city"]=all_data["Purchase Address"].str.split(",",expand=True)[1]

results = all_data.groupby("city").sum()

results["city"]
cities=[]
