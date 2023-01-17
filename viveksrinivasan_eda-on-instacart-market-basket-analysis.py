import pylab

import calendar

import numpy as np

import pandas as pd

import seaborn as sn

from scipy import stats

import missingno as msno

from datetime import datetime

import matplotlib.pyplot as plt

import warnings

pd.options.mode.chained_assignment = None

warnings.filterwarnings("ignore", category=DeprecationWarning)

%matplotlib inline

sn.set_palette(palette="OrRd")
aisles = pd.read_csv('../input/aisles.csv')

departments = pd.read_csv('../input//departments.csv')

orderProductsTrain = pd.read_csv('../input/order_products__train.csv')

orders = pd.read_csv('../input/orders.csv')

products = pd.read_csv('../input/products.csv')

orderProductsPrior = pd.read_csv('../input/order_products__prior.csv')
fig,ax = plt.subplots()

fig.set_size_inches(20,5)

sn.countplot(data=orders,x="order_hour_of_day",ax=ax,color="#34495e")

ax.set(xlabel='Hour Of The Day',title="Order Count Across Hour Of The Day")
fig,ax = plt.subplots()

fig.set_size_inches(20,5)

ordersDay = orders[["order_dow"]].replace({0:"Sunday",1:"Monday",2:"Tuesday",3:"Wednesday",4:"Thursday",5:"Friday",6:"Saturday"})



sn.countplot(color="#34495e",data=ordersDay,x="order_dow",ax=ax,order=["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"])

ax.set(xlabel='Day Of The Week',title="Order Count Across Days Of The Week")
fig,ax = plt.subplots()

fig.set_size_inches(20,5)

sn.countplot(color="#34495e",data=orders,x="days_since_prior_order",ax=ax)

ax.set(xlabel='Hour Of The Day',title="Reorder Count")
orderCount = orders[orders["eval_set"]=="prior"].groupby(by=["user_id"])["order_id"].count().to_frame()

fig,ax = plt.subplots()

fig.set_size_inches(20,5)

sn.countplot(color="#34495e",data=orderCount,x="order_id",ax=ax)

ax.set(xlabel='Order Count',title="Order Count")
fig,(ax1,ax2) = plt.subplots(nrows=2)

fig.set_size_inches(20,15)

orderCountsPrior = orderProductsPrior.groupby("order_id")["product_id"].count().to_frame()["product_id"].value_counts().to_frame()

orderCountsPrior["count"] = orderCountsPrior["product_id"]

orderCountsPrior["no_of_products"] = orderCountsPrior.index

orderCountsTrain = orderProductsTrain.groupby("order_id")["product_id"].count().to_frame()["product_id"].value_counts().to_frame()

orderCountsTrain["count"] = orderCountsTrain["product_id"]

orderCountsTrain["no_of_products"] = orderCountsTrain.index

sn.barplot(data=orderCountsTrain.head(50),x="no_of_products",y="count",ax=ax1,color="#34495e")

sn.barplot(data=orderCountsPrior.head(50),x="no_of_products",y="count",ax=ax2,color="#34495e")

ax1.set(xlabel='Order Count',title="Count Of Items People Buy In Train")

ax2.set(xlabel='Order Count',title="Count Of Items People Buy In Prior")
productsCount = orderProductsTrain["product_id"].value_counts().to_frame()

productsCount["count"] = productsCount.product_id

productsCount["product_id"] = productsCount.index

mergedData = pd.merge(productsCount,products,how="left",on="product_id").sort_values(by="count",ascending=False)



fig,ax = plt.subplots()

fig.set_size_inches(25,10)

sn.barplot(data=mergedData.head(30),x="product_name",y="count",ax=ax,orient="v",color="#34495e")

ax.set(xlabel='Product Names',ylabel="Count",title="Best Selling Products")

plt.xticks(rotation=90)



mergedData.head(10)
productsCountReordered = orderProductsTrain[orderProductsTrain["reordered"]==1]["product_id"].value_counts().to_frame()

productsCountReordered["reordered_count"] = productsCountReordered.product_id

productsCountReordered["product_id"] = productsCountReordered.index

productCountReorderedMerged = pd.merge(productsCount,productsCountReordered,how="left",on="product_id").sort_values(by="count",ascending=False)

productCountReorderedMerged["reordered_ratio"] = productCountReorderedMerged["reordered_count"]/productCountReorderedMerged["count"]

productCountReorderedMerged.sort_values(by="reordered_ratio",ascending=False,inplace=True)

productMerged = pd.merge(productCountReorderedMerged,products,how="left",on="product_id")



fig,ax = plt.subplots()

fig.set_size_inches(25,10)

sn.barplot(data=productMerged[productMerged["count"]>40].head(30),x="product_name",y="reordered_ratio",color="#34495e",ax=ax,orient="v")

ax.set(xlabel='Product Names',ylabel="Count",title="Top Reordered Products")

ax.set_ylim(0.85,.95)

plt.xticks(rotation=90)



productMerged.head(10)
productsCountFirst = orderProductsTrain[orderProductsTrain["add_to_cart_order"]==1]["product_id"].value_counts().to_frame()

productsCountFirst["reordered_count"] = productsCountFirst.product_id

productsCountFirst["product_id"] = productsCountFirst.index

productCountFirstMerged = pd.merge(productsCount,productsCountFirst,how="left",on="product_id").sort_values(by="count",ascending=False)

productCountFirstMerged["first_ordered_ratio"] = productCountFirstMerged["reordered_count"]/productCountFirstMerged["count"]

productCountFirstMerged.sort_values(by="first_ordered_ratio",ascending=False,inplace=True)

firstMerged = pd.merge(productCountFirstMerged,products,how="left",on="product_id")





fig,ax = plt.subplots()

fig.set_size_inches(25,10)

sn.barplot(data=firstMerged[firstMerged["count"]>10].head(30),x="product_name",y="first_ordered_ratio",color="#34495e",ax=ax,orient="v")

ax.set(xlabel='Product Names',ylabel="Count",title="Top Reordered Products")

ax.set_ylim(0.4,.7)

plt.xticks(rotation=90)



firstMerged.head(10)