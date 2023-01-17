import numpy as np

import pandas as pd 

import os

import seaborn as sns; sns.set(rc={'figure.figsize':(16,9)})

import matplotlib.pyplot as plt





geo = pd.read_csv("../input/olist_geolocation_dataset.csv", dtype={'geolocation_zip_code_prefix': str})





# tratamento do campo CEP, para posteriores consultas

geo['geolocation_zip_code_prefix_1_digits'] = geo['geolocation_zip_code_prefix'].str[0:1]

geo['geolocation_zip_code_prefix_2_digits'] = geo['geolocation_zip_code_prefix'].str[0:2]

geo['geolocation_zip_code_prefix_3_digits'] = geo['geolocation_zip_code_prefix'].str[0:3]

geo['geolocation_zip_code_prefix_4_digits'] = geo['geolocation_zip_code_prefix'].str[0:4]

geo['geolocation_zip_code_prefix_5_digits'] = geo['geolocation_zip_code_prefix'].str[0:5]





# exclusão de ordens realizadas fora do território brasileiro

geo = geo[geo.geolocation_lat <= 5.27438888]

geo = geo[geo.geolocation_lng >= -73.98283055]

geo = geo[geo.geolocation_lat >= -33.75116944]

geo = geo[geo.geolocation_lng <=  -34.79314722]



# conversão de coordenadas para Mercator

from datashader.utils import lnglat_to_meters as webm

x, y = webm(geo.geolocation_lng, geo.geolocation_lat)

geo['x'] = pd.Series(x)

geo['y'] = pd.Series(y)
geo.head(10)
# correção dos CEP's para plotagem 

geo['geolocation_zip_code_prefix'] = geo['geolocation_zip_code_prefix'].astype(int)

geo['geolocation_zip_code_prefix_1_digits'] = geo['geolocation_zip_code_prefix_1_digits'].astype(int)

geo['geolocation_zip_code_prefix_2_digits'] = geo['geolocation_zip_code_prefix_2_digits'].astype(int)

geo['geolocation_zip_code_prefix_3_digits'] = geo['geolocation_zip_code_prefix_3_digits'].astype(int)

geo['geolocation_zip_code_prefix_4_digits'] = geo['geolocation_zip_code_prefix_4_digits'].astype(int)

geo['geolocation_zip_code_prefix_5_digits'] = geo['geolocation_zip_code_prefix_5_digits'].astype(int)



brazil = geo

agg_name = 'geolocation_zip_code_prefix'

#brazil[agg_name].describe().to_frame()
# plot wtih holoviews + datashader - bokeh with map background

import holoviews as hv

import geoviews as gv

import datashader as ds

from colorcet import fire, rainbow, bgy, bjy, bkr, kb, kr

from datashader.colors import colormap_select, Greys9

from holoviews.streams import RangeXY

from holoviews.operation.datashader import datashade, dynspread, rasterize

from bokeh.io import push_notebook, show, output_notebook

output_notebook()

hv.extension('bokeh')



%opts Overlay [width=800 height=600 toolbar='above' xaxis=None yaxis=None]

%opts QuadMesh [tools=['hover'] colorbar=True] (alpha=0 hover_alpha=0.2)



T = 0.05

PX = 1



def plot_map(data, label, agg_data, agg_name, cmap):

    url="http://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{Z}/{Y}/{X}.png"

    geomap = gv.WMTS(url)

    points = hv.Points(gv.Dataset(data, kdims=['x', 'y'], vdims=[agg_name]))

    agg = datashade(points, element_type=gv.Image, aggregator=agg_data, cmap=cmap)

    zip_codes = dynspread(agg, threshold=T, max_px=PX)

    hover = hv.util.Dynamic(rasterize(points, aggregator=agg_data, width=50, height=25, streams=[RangeXY]), operation=hv.QuadMesh)

    hover = hover.options(cmap=cmap)

    img = geomap * zip_codes * hover

    img = img.relabel(label)

    return img



# plot wtih datashader - image with black background

import datashader as ds

from datashader import transfer_functions as tf

from functools import partial

from datashader.utils import export_image

from IPython.core.display import HTML, display

from colorcet import fire, rainbow, bgy, bjy, bkr, kb, kr



background = "black"

cm = partial(colormap_select, reverse=(background!="black"))

export = partial(export_image, background = background, export_path="export")

display(HTML("<style>.container { width:100% !important; }</style>"))

W = 700 



def create_map(data, cmap, data_agg, export_name='img'):

    pad = (data.x.max() - data.x.min())/50

    x_range, y_range = ((data.x.min() - pad, data.x.max() + pad), 

                             (data.y.min() - pad, data.y.max() + pad))



    ratio = (y_range[1] - y_range[0]) / (x_range[1] - x_range[0])



    plot_width  = int(W)

    plot_height = int(plot_width * ratio)

    if ratio > 1.5:

        plot_height = 550

        plot_width = int(plot_height / ratio)

        

    cvs = ds.Canvas(plot_width=plot_width, plot_height=plot_height, x_range=x_range, y_range=y_range)



    agg = cvs.points(data, 'x', 'y', data_agg)

    img = tf.shade(agg, cmap=cmap, how='eq_hist')

    return export(img, export_name)



def filter_data(level, name):

    df = geo[geo[level] == name]

    #remove outliers

    df = df[(df.x <= df.x.quantile(0.999)) & (df.x >= df.x.quantile(0.001))]

    df = df[(df.y <= df.y.quantile(0.999)) & (df.y >= df.y.quantile(0.001))]

    return df
americana = geo[geo['geolocation_city'] == 'americana']

agg_name = 'geolocation_zip_code_prefix'





plot_map(americana, 'CEPs que realizaram compras em Americana', ds.min(agg_name), agg_name, cmap=rainbow)

orders_df = pd.read_csv('../input/olist_orders_dataset.csv')

order_items = pd.read_csv('../input/olist_order_items_dataset.csv')

order_reviews = pd.read_csv('../input/olist_order_reviews_dataset.csv')

customer = pd.read_csv('../input/olist_customers_dataset.csv', dtype={'customer_zip_code_prefix': str})



# getting the first 3 digits of customer zipcode

customer['customer_zip_code_prefix_3_digits'] = customer['customer_zip_code_prefix'].str[0:3]

customer['customer_zip_code_prefix_3_digits'] = customer['customer_zip_code_prefix_3_digits'].astype(int)



brazil_geo = geo.set_index('geolocation_zip_code_prefix_3_digits').copy()



orders = orders_df.merge(order_items, on='order_id')

orders = orders.merge(customer, on='customer_id')

orders = orders.merge(order_reviews, on='order_id')



gp = orders.groupby('customer_zip_code_prefix_3_digits')['price'].sum().to_frame()

saopaulo = filter_data('geolocation_state', 'SP').set_index('geolocation_zip_code_prefix_3_digits')

revenue = saopaulo.join(gp)

agg_name = 'revenue'

revenue[agg_name] = revenue.price / 1000
plot_map(revenue, 'Receita dos pedidos (R$ 1000,00)', ds.mean(agg_name), agg_name, cmap=fire)
create_map(revenue, fire, ds.mean(agg_name), 'revenue_brazil')
import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import matplotlib.dates as mdates

%matplotlib inline 

import seaborn as sns

import datetime

import scipy.stats as stats
customers = pd.read_csv("../input/olist_customers_dataset.csv")

geoloc = pd.read_csv("../input/olist_geolocation_dataset.csv")

items = pd.read_csv("../input/olist_order_items_dataset.csv")

payments = pd.read_csv("../input/olist_order_payments_dataset.csv")

reviews = pd.read_csv("../input/olist_order_reviews_dataset.csv")

orders = pd.read_csv("../input/olist_orders_dataset.csv")
cities = customers["customer_city"].nunique()

c1 = customers.groupby('customer_city')['customer_id'].nunique().sort_values(ascending=False)

print("Temos ",cities," cidades no dataset. As 5 cidades com mais ordens são:")

c2 = c1.head(5)

print(c2)

print("\nAs 5 maiores cidades com ordens representam", round(c2.sum()/customers.shape[0]*100,1),"% de todas as ordens")

plt.figure(figsize=(16,8))

c2.plot(kind="bar",rot=0)
fig, ax = plt.subplots(figsize=(9, 8), subplot_kw=dict(aspect="equal"))

explode = (0.1, 0, 0, 0)

colors = ['#f45a5a', '#449dfc', '#93f96d', '#f9c86d']

legend = ["Cartão de Crédito", "Boleto", "Voucher", "Débito"]



p = payments["payment_type"][payments["payment_type"] != "not_defined"].value_counts()

p.plot(kind="pie", legend=False, labels=None, startangle=0, explode=explode, autopct='%1.0f%%', pctdistance=0.6, shadow=True, textprops={'weight':'bold', 'fontsize':16}, 

       colors=colors, ax=ax)

ax.legend(legend, loc='best', shadow=True, prop={'weight':'bold', 'size':12}, bbox_to_anchor=(0.6, 0, 0.5,1))

plt.title("Forma de Pagamento", fontweight='bold', size=16)

plt.ylabel("")