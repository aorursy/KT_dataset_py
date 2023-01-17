import numpy as np 

import pandas as pd 

import plotly.express as px

import plotly.graph_objects as go
categories = pd.read_csv('/kaggle/input/summer-products-and-sales-in-ecommerce-wish/unique-categories.csv')

products = pd.read_csv('/kaggle/input/summer-products-and-sales-in-ecommerce-wish/summer-products-with-rating-and-performance_2020-08.csv')

sorted_cats = pd.read_csv('/kaggle/input/summer-products-and-sales-in-ecommerce-wish/unique-categories.sorted-by-count.csv')
products.head()
products.info()
products.isnull().sum()
px.histogram(products['rating'],nbins=20)
fig = px.histogram(products, x="rating",

                   title='Histogram of ratings',

                   labels={'rating':'total rating'}, # can specify one label per df column

                   opacity=0.8,

                   log_y=False, # represent bars with log scale

                   color_discrete_sequence=['indianred'] # color of histogram bars

                   )

fig.show()
fig = px.histogram(products, x="price", color="uses_ad_boosts",nbins=45)

fig.show()
fig = px.histogram(products, x="price", color="uses_ad_boosts", marginal="violin",nbins=40,

                         hover_data=products.columns)

fig.show()
px.bar(list(products['uses_ad_boosts'].value_counts()),[0,1],list(products['uses_ad_boosts'].value_counts()))
px.bar(sorted_cats,sorted_cats['keyword'][:10],sorted_cats['count'][:10])
px.bar(sorted_cats,sorted_cats['keyword'][:10],sorted_cats['count'][:10],color=products['uses_ad_boosts'][:10])
px.box(products['rating'])
px.box(products['retail_price'],points="all")
px.box(products['retail_price'],points="outliers")
fig = px.box(products, x="price", color="uses_ad_boosts")

fig.update_traces(quartilemethod="exclusive") # or "inclusive", or "linear" by default

fig.show()
fig = px.box(products, x="rating",  color="uses_ad_boosts",

             notched=True, # used notched shape

            )

fig.show()
px.violin(products['units_sold'])
px.violin(products['retail_price'],points="outliers")
fig = px.violin(products, x="rating",  color="uses_ad_boosts",

            )

fig.show()
px.line(products,list(range(len(products['units_sold']))),products['units_sold'])
px.line(products,list(range(len(products['units_sold']))),products['units_sold'],color=products['uses_ad_boosts'])