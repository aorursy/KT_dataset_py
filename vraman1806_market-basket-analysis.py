# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from mlxtend.frequent_patterns import apriori, association_rules

import plotly.express as px



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
market_basket_optimisation = pd.read_csv("../input/Market_Basket_Optimisation.csv", header = None)

market_basket_optimisation.head()
# market_basket_list = []

# for i in range(len(market_basket_optimisation)):

#     market_basket_list.append([str(market_basket_optimisation.values[i,j]) for j in range(len(market_basket_optimisation.columns))])

    

# len(market_basket_list)
market_items = (market_basket_optimisation[0].unique())

print(market_items)

print('No. of items:', len(market_items))
encoded_values = []

for index, row in market_basket_optimisation.iterrows():

    market_basket_labels = {}

    uncommons = list(set(market_items) - set(row))

    commons = list(set(market_items).intersection(row))

    for uc in uncommons:

        market_basket_labels[uc] = 0

    for common in commons:

        market_basket_labels[common] = 1

    encoded_values.append(market_basket_labels)



market_basket_optimisation_encoded = pd.DataFrame(encoded_values)
market_basket_optimisation_encoded.head()
item_list = market_basket_optimisation_encoded.columns

item_count = list()

for item in item_list:

        item_count.append(len(market_basket_optimisation_encoded[market_basket_optimisation_encoded[item]==1]))

    

print(item_count)
item_count_df = pd.DataFrame()

item_count_df['item'] = item_list

item_count_df['count'] = item_count



px.bar(item_count_df.sort_values(by = 'count', ascending = False).iloc[0:10,], x = 'item', y = 'count')
market_basket_frequent_items = apriori(market_basket_optimisation_encoded, min_support=0.02, use_colnames = True, verbose = 1, max_len=2)

market_basket_frequent_items['length'] = market_basket_frequent_items['itemsets'].apply(lambda x: len(x))

print(market_basket_frequent_items[market_basket_frequent_items.length==2])
market_basket_rules = association_rules(market_basket_frequent_items, metric='support', min_threshold=0.03)

market_basket_rules
market_basket_optimisation_encoded.drop(columns=['mineral water','chocolate'], inplace=True)
market_basket_frequent_items = apriori(market_basket_optimisation_encoded, min_support=0.02, use_colnames = True, verbose = 1, max_len=2)

market_basket_frequent_items['length'] = market_basket_frequent_items['itemsets'].apply(lambda x: len(x))

print(market_basket_frequent_items[market_basket_frequent_items.length==2])
market_basket_rules = association_rules(market_basket_frequent_items, metric='support', min_threshold=0.03)

market_basket_rules
market_basket_rules.drop_duplicates(['support'], inplace=True)

market_basket_rules.sort_values(by='support', ascending=False)