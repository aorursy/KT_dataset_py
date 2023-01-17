## Recommender System : Market Basket analysis using MLExtend Library.
import pandas as pd
import numpy as np
# import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori,association_rules
import matplotlib.pyplot as plt
plt.style.use('default')
 

## Load data 


data = pd.read_csv("../input/market-basket-optimization/Market_Basket_Optimisation.csv" , header = None ,engine='python')
data .head()
data.shape
##  capture just row lables by acessing the index in shape

data[1]
## Lets do the EDA to Understand the DAta Better, 

# 1.  Get all the items in the dataset, to get the total number of items.

"""Create an empty list to gather the values, and convert in np.array"""
items = []

"""Loop through each value in rows and columns to get each value in the list, no matter even if they are repeated."""
for i in range(0, data.shape[0]):
    for j in range(0, data.shape[1]):
        items.append(data.values[i,j])

items = np.array(items)
print("Total Number of items present in the dataset",len(items))
df = pd.DataFrame(items, columns = ['items'])
df['items'].value_counts().head(10)
df['items'].unique()
## get the Count of the unique values in items by grouping them.

"""Make a New column in the dataset , by putting a value of digit 1, so that we can add them by groupping the similar values."""

df['item_count'] = 1
df.shape
## drop the nan values from the dataset as it will not represent any kind of transaction

nan_drop = df[df['items'].isnull()].index
df.drop(nan_drop , inplace = True)
df.shape
## create items list by ascending order to view which one is occuring most

df_items_list = df.groupby(['items']).sum().reset_index().sort_values(by = 'item_count' , ascending = False)
df_items_list.head(10).style.background_gradient(cmap='BuPu')
"""USing TransactionEncoder method to encode the values"""

"""Create an Empty List, loop through the items in rows, and then all the columns, and convert them in 0 or 1"""
items = []
for i in range(data.shape[0]):
    items.append([str(data.values[i,j]) for j in range(data.shape[1])])
    
    
items = np.array(items)
tr = TransactionEncoder()
tr_Array = tr.fit(items).transform(items)

df = pd.DataFrame(tr_Array , columns = tr.columns_)
df
# Convert dataset into 1-0 encoding

def encode_units(x):
    if x == False:
        return 0 
    if x == True:
        return 1
    
df = df.applymap(encode_units)
df.head(10)
## Support ##Confidence ## LIFT   

# Extracting the most frequest itemsets via Mlxtend.
# The length column has been added to increase ease of filtering.

frequent_itemsets = apriori(df, min_support=0.01, use_colnames=True)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
frequent_itemsets.sort_values(by = 'length' , ascending = False).head(10)
frequent_itemsets[ (frequent_itemsets['length'] == 2) &
                   (frequent_itemsets['support'] >= 0.05) ].head()
## Definging the Rules for the Algorithm

rules = association_rules(frequent_itemsets , metric='lift'  ,min_threshold= 1.2)

rules["ante._count"] = rules["antecedents"].apply(lambda x: len(x))
rules["cont._count"] = rules["consequents"].apply(lambda x: len(x))
rules.sort_values(by = "lift" , ascending =False)
# Sort values based on confidence

rules.sort_values("confidence",ascending=False)
rules[~rules['antecedents'].str.contains("mineral water" , regex=False) & 
      ~rules['consequents'].str.contains("mineral water", regex=False)].sort_values("confidence" , ascending = False)
rules[rules["antecedents"].str.contains("ground beef", regex=False) &
      rules["ante._count"] == 1].sort_values("confidence", ascending=False).head(10)
# For optimum results, you can filter out both the metrics and filter out the most relevant associated items.

rules[(rules['lift'] >= 2) & (rules['confidence'] >= 0.50)]
# Results












































































































































