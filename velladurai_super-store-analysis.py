# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np# linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np 

import pandas as pd

import os

import matplotlib.pyplot as plt

import seaborn as sns
print(os.getcwd())

os.listdir()

os.getcwd()

pd.options.display.max_columns = 300
df = pd.read_csv("../input/superstore_dataset2011-2015.csv", encoding = "ISO-8859-1")
df.shape

df.head()

df.columns

df.dtypes
result = df.groupby(["Customer Name"])['Profit'].aggregate(np.sum).reset_index().sort_values('Profit',ascending = False).head(20)

result.head

type(result)

result.shape

result

sns.barplot(x = "Customer Name",y= "Profit",data=result)
sortedTop20 = df.sort_values(['Profit'], ascending=False).head(20)

fig = plt.figure(figsize=(12,8))

ax = fig.add_subplot(111)

p = sns.barplot(x='Customer Name', y='Profit', data=sortedTop20, ax=ax)

ax.set_title("Top 20 profitable Customers")

ax.set_xticklabels(p.get_xticklabels(), rotation=75)

plt.tight_layout()

plt.show()
fig = plt.figure(figsize = (5,5))
ax1 = fig.add_subplot(111)

sns.barplot(x = "Customer Name",y= "Profit",

            data=result,

            ax = ax1

             )
ax1.set_ylabel("Profit", fontname="Arial", fontsize=12)

# Set the title to Comic Sans

ax1.set_title("Top 20 Customers", fontname='Comic Sans MS', fontsize=18)

# Set the font name for axis tick labels to be Comic Sans

for tick in ax1.get_xticklabels():

    tick.set_fontname("Comic Sans MS")

    tick.set_fontsize(12)

for tick in ax1.get_yticklabels():

    tick.set_fontname("Comic Sans MS")

    tick.set_fontsize(12)
# Rotate the labels as the Customer names overwrites on top of each other

ax1.set_xticklabels(ax1.get_xticklabels(), rotation = 45)

plt.show()

#---Observations

## The top 3 customers in that order are Tamara Chand, Raymond Buch & Sanjit Chand
#=================================================================================

# 2. What is the distribution of our customer segment

#=================================================================================

descending_order = df['Segment'].value_counts().index



df.Segment.value_counts()
#What is the distribution of our customer segment

fig = plt.figure(figsize=(12,8))

ax = fig.add_subplot(111)

p = sns.countplot(x="Segment", data=df, ax=ax)

ax.set_title("Customer Distribution by Segment")

ax.set_xticklabels(p.get_xticklabels(), rotation=90)

for p in ax.patches:

    ax.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.15, p.get_height()+1))

plt.tight_layout()

plt.show()
#---Observations

# Segment is categorical attribute with 3 levels - Consumer, Corporate & Home Office. The distribution is highest in Consumer

# followed by Corporate and Home Office



#3. Who are our top-20 oldest customers

oldCustomers = df.sort_values(["Order Date"], ascending=True).iloc[0:20,6]

oldCustomers
# Top oldest customers are Annie Thurman, Eugene Moren, Joseph Holt, Toby Braunhardt, Dave Hallsten



#==========================================================================================

#5. Relationship of Order Priority and Profit

df['Order Priority'].value_counts()



sns.boxplot(

            "Order Priority",

            "Profit",

             data= df

             )
#there does not appear to be any relationship between Order Priority & Profit

#6. What is the distribution of customers Market wise?

df.shape

df['Market'].value_counts()

Customers_market = pd.DataFrame({'Count' : df.groupby(["Market","Customer Name"]).size()}).reset_index()

Customers_market.shape

sns.barplot(x = "Market",     # Data is groupedby this variable

             y= "Count",    # Aggregated by this variable

             data=Customers_market

             )
sns.countplot("Market",        # Variable whose distribution is of interest

                data = Customers_market)
# Market has 7 levels. APAC has the largest # of customers followed by LATAM, and US in that order

 # Canada has the least # of customers

    #7. What is the distribution of customers Market wise and Region wise

df['Region'].value_counts()

Customers_market_region = pd.DataFrame({'Count' : df.groupby(["Market","Region","Customer Name"]).size()}).reset_index()



sns.countplot("Market",        # Variable whose distribution is of interest

              hue= "Region",    # Distribution will be gender-wise

              data = Customers_market_region)
#for APAC, the largest # of customers are basd out of Oceania, followed by Southeast Asia

#for US, the largest # of customers are based out of Western Region followed by East

#8.Distribution of  Customers by Country & State - top 15

Customers_Country = pd.DataFrame({'Count' : df.groupby(["Country","State"]).size()}).reset_index().sort_values('Count',ascending = False).head(15)

Customers_Country



sns.barplot(x = "Country",     # Data is groupedby this variable

            y= "Count",  

            hue="State",

            data = Customers_Country.sort_values('Country')

            )
## US has the largest number of customers -California being the largest followed by New York, Washington, Illinois & Ohio

## UK has the next largest population of Customers -England



# Top 20 Cities by Sales Volume

sale_cities = df.groupby(["City"])['Quantity'].aggregate(np.sum).reset_index().sort_values('Quantity',ascending = False).head(20)

sns.barplot(x = "City",     # Data is groupedby this variable

            y= "Quantity",          

            data=sale_cities,

            )

# top 10 products

sale_Products = df.groupby(["Product Name"])['Quantity'].aggregate(np.sum).reset_index().sort_values('Quantity',ascending = False).head(20)

sns.barplot(x = "Product Name",     # Data is groupedby this variable

            y= "Quantity",          

            data=sale_Products)
#Staples is the largest selling product



# top selling products by countries (in US)

df.columns

sale_Products_Country = df.groupby(["Product Name","Country"])['Quantity'].aggregate(np.sum).reset_index().sort_values('Quantity',ascending = False)

sale_Products_Country = df.groupby(["Product Name","Country"])['Quantity'].sum().reset_index().sort_values('Quantity',ascending = False)

sale_Products_Country

type(sale_Products_Country)

spc = sale_Products_Country[sale_Products_Country['Country'] == "United States"].sort_values('Quantity',ascending = False).head(20)

sns.barplot(x = "Product Name",     # Data is groupedby this variable

            hue="Country",

            y= "Quantity",          

            data=spc)

# sales by product Category, Sub-category

sale_category = df.groupby(["Category","Sub-Category"])['Quantity'].aggregate(np.sum).reset_index().sort_values('Quantity',ascending = False)

sale_category

sns.barplot(x = "Category",     # Data is groupedby this variable

            hue="Sub-Category",

            y= "Quantity",          

            data=sale_category)
#Year-wise sales and profit

df["Order_Year"] = pd.to_datetime(df["Order Date"])

df["Year"] = df["Order_Year"].dt.year

yearwiseSalesAndProfit = df.groupby("Year").agg({"Sales":np.sum, "Profit": np.sum})
yearwiseSalesAndProfit
fig = plt.figure(figsize=(16,8))

ax = fig.add_subplot(221)

p=sns.barplot(x=yearwiseSalesAndProfit.index,y="Profit", data=yearwiseSalesAndProfit, palette="winter", ax=ax)

ax.set_title("Year-wise Profit")

ax.set_xticklabels(p.get_xticklabels(), rotation=0)

ax = fig.add_subplot(222)

p=sns.barplot(x=yearwiseSalesAndProfit.index,y="Sales", data=yearwiseSalesAndProfit, palette="spring", ax=ax)

ax.set_title("Year-wise Sales")

ax.set_xticklabels(p.get_xticklabels(), rotation=0)

plt.tight_layout()

plt.show()
#Relationship between sales and profit -- use scatter plot

regionwiseSalesAndProfit = df.groupby("Region").agg({"Sales":np.sum, "Profit": np.sum})

regionwiseSalesAndProfit

fig = plt.figure(figsize=(12,8))

ax = fig.add_subplot(111)

p = sns.scatterplot(x="Sales", y="Profit", hue=regionwiseSalesAndProfit.index, data=regionwiseSalesAndProfit) # kind="scatter")

ax.set_title("Relationship between Sales and Profit by Region")

plt.tight_layout()

plt.show()