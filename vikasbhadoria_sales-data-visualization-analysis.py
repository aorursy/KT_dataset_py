import pandas as pd
%matplotlib inline
data=pd.read_csv("../input/final-dataset/all_data.csv")
data.head()
nan_df=data[data.isna().any(axis=1)]
nan_df.head()
data.info()
data=data.dropna(how="all")
data.head()
data.tail()
data_null=data[data.isnull().any(axis=1)]

data["Product"].value_counts()
data = data[data["Order Date"].str[0:2]!="Or"]
data.head()
data["Month"] = pd.to_datetime(data["Order Date"]).dt.month
data.head()
def get_city(address):
    return address.split(",")[1].strip()

def get_state(address):
    return address.split(",")[2].strip(" ")[0:2]

data['all_cities']= data["Purchase Address"].apply(lambda x: f"{get_city(x)}({get_state(x)})")
data['Quantity Ordered'] = pd.to_numeric(data['Quantity Ordered'])
data['Price Each'] = pd.to_numeric(data['Price Each'])
data.head()
data["Month"].value_counts()
data["Sales"] = data["Quantity Ordered"]*data["Price Each"]
month_wise_sale=data.groupby(["Month"]).sum()
month_wise_sale
import matplotlib.pyplot as plt
months=range(1,13)

plt.figure(figsize=(12,8))
plt.bar(months,month_wise_sale['Sales'])
plt.xticks(months)
plt.xlabel("Month Number")
plt.ylabel("Sales in USD")
plt.show()
city_wise_sale=data.groupby(["all_cities"]).sum()
city_wise_sale
keys = [city for city, df in data.groupby(['all_cities'])]

plt.figure(figsize=(12,8))
plt.bar(keys,city_wise_sale['Sales'])
plt.xticks(keys,rotation="vertical",size=15)
plt.xlabel("All Cities")
plt.ylabel("Sales in USD")
plt.show()
data["Hour"] = pd.to_datetime(data["Order Date"]).dt.hour
data["Minute"] = pd.to_datetime(data["Order Date"]).dt.minute
data["Count"]=1

data.head()
keys=range(24)
plt.plot(keys, data.groupby(['Hour']).count()['Count'])
plt.xticks(keys)
plt.grid()
plt.show()

# My recommendation is slightly before 11am or 7pm
