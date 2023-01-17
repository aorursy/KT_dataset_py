# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Set Options

pd.options.display.max_columns

pd.options.display.max_rows = 500
#load Data

store_data =pd.read_excel("/kaggle/input/uci-online-retail-ii-data-set/online_retail_II.xlsx",sheet_name= "Year 2010-2011" )
#Get copy of dataFrame

df = store_data.copy()
#del df
#Get copy of dataFrame

df = store_data.copy()
df
# Find How many different countries are there?

df["Country"].nunique()
#List countries as unique?

df["Country"].unique()
# List product as Unique?

df["Description"].unique()
# Check Customers according to Country



pd.DataFrame(df["Country"].value_counts(normalize=True)).head(100)
# Display how many products are there as by groping?  

df["Description"].value_counts()
# List most ordered products by ascending 

df.groupby("Description").agg({"Quantity":"sum"}).sort_values(by='Quantity', ascending=True).head()
#Add a column for total price to calculate monetary attritube

df["TotalPrice"] = df["Quantity"]*df["Price"]
df
#Find total price for each Invoice by goruping

df.groupby("Invoice").agg({"TotalPrice": "sum"}).head()
#Check Missing values for each Column
(df.isnull().sum())
# Check Total Missing Values
(df.isnull().sum()).sum()
#Remove missing Values From CustomerID
df.dropna(subset= ["Customer ID"],inplace= True)
df["Customer ID"].isnull().sum()
df
#Remove negative values from Quantity Column
deleteRows = df[~df['Quantity'] > 0].index

df = df.drop(deleteRows, axis=0)
df
#Some rows start with C means refund so we will remove them

deleteRows =  df[df["Invoice"].str.contains("C", na=False)].index

df = df.drop(deleteRows, axis=0)
#Remove POSTAGE

deleteRows =  df[df["Description"].str.contains("POSTAGE", na=False)].index

df = df.drop(deleteRows, axis=0)
df
df.head()
#Find out the first and last order dates in the data.

df["InvoiceDate"].min()
df['InvoiceDate'].max()
#Since recency is calculated for a point in time, and the last invoice date is 2011–12–09, we will use 2011–12–10 to calculate recency.



import datetime as dt

today_date = dt.datetime(2011,12,9)
recency_df = df.groupby("Customer ID").agg({'InvoiceDate': lambda x: (today_date - x.max()).days})
recency_df.rename(columns={"InvoiceDate":"Recency"}, inplace= True)
recency_df
#temp_df= df.groupby(['Customer ID']).agg({'Invoice': "count"})

temp_df =  df.groupby(['Customer ID','Invoice']).agg({'Invoice': "count"}).groupby(['Customer ID']).agg({"Invoice": "count"})
temp_df
freq_df = temp_df.rename(columns={"Invoice": "Frequency"})

freq_df
monetary_df=df.groupby("Customer ID").agg({'TotalPrice': "sum"})
monetary_df
monetary_df.rename(columns={"TotalPrice": "Monetary"}, inplace=True)
monetary_df
rfm = pd.concat([recency_df,freq_df,monetary_df], axis = 1)
rfm.head(500)
#First Customer has frequency 1 monetary value 77183.60 and recency 325
# Get RFM scores for 3 attribute

rfm["RecencyScore"] = pd.qcut(rfm['Recency'],5, labels=[5,4,3,2,1])
#if you calculate only transaction operations(unique invoice per customer) add rank(method="first")

#iy you sum all operations in per invoice no need to add rank method

rfm["FrequencyScore"] = pd.qcut(rfm['Frequency'].rank(method="first"),5, labels=[1,2,3,4,5])
rfm["MonetaryScore"] = pd.qcut(rfm['Monetary'],5, labels=[1,2,3,4,5])
rfm.head()
rfm["RFM_SCORE"] = rfm["RecencyScore"].astype(str) +rfm["FrequencyScore"].astype(str)+rfm["MonetaryScore"].astype(str) 
rfm.head()
rfm.describe().T
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
rfm["Segment"] = rfm["RecencyScore"].astype(str) + rfm["FrequencyScore"].astype(str)

rfm["Segment"] = rfm["Segment"].replace(seg_map,regex=True)

rfm.head()
rfm[["Segment","Recency","Frequency","Monetary"]].groupby("Segment").agg(["mean","count"])
rfm[rfm["Segment"] == "Need Attention"].head()
rfm[rfm["Segment"] == "New Customers"].index
new_df = pd.DataFrame()

new_df["NewCustomerID"] = rfm[rfm["Segment"] == "New Customers"].index
new_df.head(100)
segments_counts = rfm["Segment"].value_counts().sort_values(ascending=True)
segments_counts

# count the number of customers in each segment

segments_counts = rfm['Segment'].value_counts().sort_values(ascending=True)



fig, ax = plt.subplots()



bars = ax.barh(range(len(segments_counts)),

              segments_counts,

              color='silver')

ax.set_frame_on(False)

ax.tick_params(left=False,

               bottom=False,

               labelbottom=False)

ax.set_yticks(range(len(segments_counts)))

ax.set_yticklabels(segments_counts.index)



for i, bar in enumerate(bars):

        value = bar.get_width()

        if segments_counts.index[i] in ['Champions', 'Loyal Customers']:

            bar.set_color('firebrick')

        ax.text(value,

                bar.get_y() + bar.get_height()/2,

                '{:,} ({:}%)'.format(int(value),

                                   int(value*100/segments_counts.sum())),

                va='center',

                ha='left',

                fontsize=14,

                  weight='bold'

               )



plt.show()
#%24 customers who dont frequently from us.

#%33 of customers are loyal and Champions customers (Loyal Customers + Champions) 

#%1 of customers are new Customers

#%13 of customers at risk 



#We have to gain customers at risk, 

#to improve the sales Potential and Best customers must be awarded
#List less Frequency of Customers by the product

import seaborn as sns

sns.set_style("whitegrid")

customer_order = df.groupby('Customer ID')['Invoice'].nunique().reset_index().head(11)

plt.figure(figsize=(10,8))

sns.barplot(data=customer_order,x="Customer ID",y="Invoice", palette="Greens_d",orient=True);

plt.xticks(rotation= 45)

plt.ylabel('Invoce Count')

plt.ylabel('Customers')
#Customers per segment



rfm_segment = rfm.reset_index()

rfm_segment.reset_index(inplace=True)





import seaborn as sns

sns.set_style("whitegrid")





customer_order = rfm_segment.groupby('Segment')['Customer ID'].nunique().reset_index().head(11)

plt.figure(figsize=(10,8))

sns.barplot(data=customer_order,x="Segment",y="Customer ID", palette="Greens_d",orient=True);

plt.xticks(rotation= 45)

plt.ylabel('Customers count')

rfm['RFM_Segment_Score'] = rfm[['RecencyScore','FrequencyScore','MonetaryScore']].sum(axis=1)
rfm['Score'] = 'Green'

rfm.loc[rfm['RFM_Segment_Score']>5,'Score'] = 'Bronze' 

rfm.loc[rfm['RFM_Segment_Score']>7,'Score'] = 'Silver' 

rfm.loc[rfm['RFM_Segment_Score']>9,'Score'] = 'Gold' 

rfm.loc[rfm['RFM_Segment_Score']>10,'Score'] = 'Platinum'
rfm
rfm_segment = rfm.reset_index()

rfm_segment.reset_index(inplace=True)



import seaborn as sns

sns.set_style("whitegrid")





customer_order = rfm_segment.groupby('Score')['Customer ID'].nunique().reset_index().head(11)

plt.figure(figsize=(10,8))

sns.barplot(data=customer_order,x="Score",y="Customer ID", palette="Greens_d",orient=True);

plt.xticks(rotation= 45)

plt.ylabel('Customers count')