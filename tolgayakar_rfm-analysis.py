#importing packages

import pandas as pd

import numpy as np

import seaborn as sns

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning) 

warnings.filterwarnings("ignore", category=FutureWarning) 

import datetime as dt

df = pd.read_excel("../input/uci-online-retail-ii-data-set/online_retail_II.xlsx" ,sheet_name= "Year 2010-2011")
df.head()
# information of the dataset

df.info()
df.isnull().sum()
df[(df["Description"].isnull()) & (df["Price"] == 0)]
df["Description"].nunique()
df["Description"].value_counts().head()
df.groupby("Description").agg({"Quantity":"sum"}).head()
df.groupby("Customer ID")["StockCode"].count()
#total unique invoice

df["Invoice"].nunique()


df.groupby("Description").agg({"Quantity":"sum"}).sort_values("Quantity", ascending = False).head()
df["TotalPrice"] = df["Quantity"] * df["Price"]
df.head()
df.isnull().sum()
df[(df["Description"].isnull()) & (df["Price"] == 0.0) & (df["Customer ID"].isnull())]
#checking missing values of pattern



import missingno as msno



msno.bar(df,figsize=(20,10));
df[df["Quantity"] < 0]
df[df["Invoice"].astype(str).map(lambda x : x.startswith("C"))]



df.dropna(inplace = True)
df["Customer ID"] = df["Customer ID"].astype(int)
df.head()

#tableau of RFM scoring



from PIL import Image



Image.open("../input/rfmimage/rfm-segments .png")

df["InvoiceDate"].max()
today_date = dt.datetime(2011,12,9)
df.groupby("Customer ID").agg({"InvoiceDate":"max"}).head()
df_ = (today_date - df.groupby("Customer ID").agg({"InvoiceDate":"max"}))
df_.rename(columns={"InvoiceDate": "Recency"}, inplace = True)
df_.head()
r_df = df_["Recency"].apply(lambda x: x.days)
r_df.head()
df_ = df.groupby(["Customer ID","Invoice"]).agg({"Invoice": "count"})
df_.head()
df_.groupby("Customer ID").agg({"Invoice":"sum"}).head()
freq_df = df_.groupby("Customer ID").agg({"Invoice":"sum"})

freq_df.rename(columns={"Invoice": "Frequency"}, inplace = True)

freq_df.head()
mntr_df = df.groupby("Customer ID").agg({"TotalPrice":"sum"})
mntr_df.head()
mntr_df.rename(columns={"TotalPrice": "Monetary"}, inplace = True)
print(r_df.shape,freq_df.shape,mntr_df.shape)
rfm = pd.concat([r_df, freq_df, mntr_df],  axis=1)
rfm.head()
df[df["Customer ID"] == 12346]
rfm["RecencyScore"] = pd.qcut(rfm['Recency'], 5, labels = [5, 4, 3, 2, 1])
rfm["FrequencyScore"] = pd.qcut(rfm['Frequency'], 5, labels = [1, 2, 3, 4, 5])
rfm["MonetaryScore"] = pd.qcut(rfm['Monetary'], 5, labels = [1, 2, 3, 4, 5])
rfm.head()
rfm["Monetary"] = rfm["Monetary"].astype(int)
rfm.head()
rfm["RFM_SCORE"] = rfm['RecencyScore'].astype(str) + rfm['FrequencyScore'].astype(str) + rfm['MonetaryScore'].astype(str)
seg_map = {

    r'[1-2][1-2]': 'Hibernating',

    r'[1-2][3-4]': 'At Risk',

    r'[1-2]5': 'Can\'t Loose',

    r'3[1-2]': 'About to Sleep',

    r'33': 'Need Attention',

    r'[3-4][4-5]': 'Loyal Customers',

    r'41': 'Promising',

    r'51': 'New Customers',

    r'[4-5][2-3]': 'Potential Loyalists',

    r'5[4-5]': 'Champions'

}
rfm['Segment'] = rfm['RecencyScore'].astype(str) + rfm['FrequencyScore'].astype(str)

rfm['Segment'] = rfm['Segment'].replace(seg_map, regex=True)

rfm.head()
from plotly.offline import init_notebook_mode, iplot, plot





color = [i for i in rfm["Segment"].value_counts()]

data = [

    {

        "y":  rfm["Segment"],

        "x": rfm["RFM_SCORE"],

        "mode": 'markers',

        "marker": {

            

            "color" : color,

            "showscale": True

        },

        "text" : rfm["RFM_SCORE"]   

    }

]

iplot(data)
rfm[rfm["Segment"] == "Need Attention"]
seg_valu = rfm["Segment"].unique().tolist()
seg_valu


explode = np.zeros(len(seg_valu))

explode[seg_valu.index("Need Attention")] = 0.2

explode
sizes = rfm["Segment"].value_counts()
import matplotlib.pyplot as plt

fig1, ax1 = plt.subplots()

ax1.pie(sizes,explode = explode,labels=seg_valu, autopct='%1.1f%%',

        shadow=True, startangle=200)

ax1.axis('equal') 



plt.show()
rfm[["Segment", "Recency","Frequency","Monetary"]].groupby("Segment").mean()
a_df = rfm[rfm["Segment"] == "Need Attention"].index

attention_df = pd.DataFrame(a_df)

attention_df.rename(columns = {"Customer ID": "NA Customer ID"},inplace= True)

attention_df.head()
#attention_df.to_excel("NAcustomerid.xlsx")