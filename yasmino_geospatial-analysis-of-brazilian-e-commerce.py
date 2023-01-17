import numpy as np
import pandas as pd 
import os

geo = pd.read_csv("../input/geolocation_olist_public_dataset.csv")
geo.head(3)
geo['zip_code_prefix'].value_counts().to_frame().describe()
# Removing some outliers
#Brazils most Northern spot is at 5 deg 16′ 27.8″ N latitude.;
geo = geo[geo.lat <= 5.27438888]
#it’s most Western spot is at 73 deg, 58′ 58.19″W Long.
geo = geo[geo.lng >= -73.98283055]
#It’s most southern spot is at 33 deg, 45′ 04.21″ S Latitude.
geo = geo[geo.lat >= -33.75116944]
#It’s most Eastern spot is 34 deg, 47′ 35.33″ W Long.
geo = geo[geo.lng <=  -34.79314722]
from datashader.utils import lnglat_to_meters as webm
x, y = webm(geo['lng'], geo['lat'])
geo['x'] = pd.Series(x)
geo['y'] = pd.Series(y)
geo.head(3)
brazil = geo
agg_name = 'zip_code_prefix'
brazil[agg_name].describe().to_frame()
# plot wtih holoviews + datashader - bokeh with map background
import holoviews as hv
import geoviews as gv
import datashader as ds
from colorcet import fire, rainbow, bgy, bjy, bkr, kb, kr
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
plot_map(brazil, 'Zip Codes in Brazil', ds.min(agg_name), agg_name, cmap=rainbow)
# plot wtih datashader - image with black background
import datashader as ds
from datashader import transfer_functions as tf
from functools import partial
from datashader.utils import export_image
from IPython.core.display import HTML, display
from colorcet import fire, rainbow, bgy, bjy, bkr, kb, kr

background = "black"
export = partial(export_image, background = background, export_path="export")
display(HTML("<style>.container { width:100% !important; }</style>"))
W = 700 

def create_map(data, cmap, data_agg):
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
    return export(img,'img')
create_map(brazil, rainbow, ds.mean(agg_name))
def filter_data(level, name):
    df = geo[geo[level] == name]
    #remove outliers
    df = df[(df.x <= df.x.quantile(0.999)) & (df.x >= df.x.quantile(0.001))]
    df = df[(df.y <= df.y.quantile(0.999)) & (df.y >= df.y.quantile(0.001))]
    return df
sp = filter_data('state', 'sp')
agg_name = 'zip_code_prefix'
sp[agg_name].describe().to_frame()
agg_name = 'zip_code_prefix'
plot_map(sp, 'Zip Codes in Sao Paulo State', ds.min(agg_name), agg_name, cmap=rainbow)
create_map(sp, rainbow, ds.mean(agg_name))
saopaulo = filter_data('city', 'sao paulo')
agg_name = 'zip_code_prefix'
saopaulo[agg_name].describe().to_frame()
plot_map(saopaulo, 'Zip Codes in Sao Paulo City', ds.min(agg_name), agg_name, cmap=rainbow)
create_map(saopaulo, rainbow, ds.mean(agg_name))
df = geo[geo['city'] == 'atibaia']
agg_name = 'zip_code_prefix'
df[agg_name].describe().to_frame()
zip129 = geo[geo[agg_name] == 129]
zip129[[agg_name, 'city', 'state']].drop_duplicates()
def plot_map2(data, label, agg_data, agg_name, cmap):
    url="http://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{Z}/{Y}/{X}.png"
    geomap = gv.WMTS(url)
    points = hv.Points(gv.Dataset(data, kdims=['x', 'y'], vdims=[agg_name]))
    agg = datashade(points, element_type=gv.Image, aggregator=agg_data, cmap=cmap)
    zip_codes = dynspread(agg, threshold=T, max_px=PX)
    img = geomap * zip_codes
    img = img.relabel(label)
    return img
plot_map2(zip129, 'Zip Codes Prefix 129', ds.min(agg_name), agg_name, cmap=rainbow)
orders = pd.read_csv('../input/olist_public_dataset_v2.csv')
brazil_geo = geo.set_index('zip_code_prefix').copy()
gp = orders.groupby('customer_zip_code_prefix')['order_products_value'].sum().to_frame()
revenue = brazil_geo.join(gp)
agg_name = 'revenue'
revenue[agg_name] = revenue.order_products_value / 1000
plot_map(revenue, 'Orders Revenue (thousands R$)', ds.mean(agg_name), agg_name, cmap=fire)
create_map(revenue, fire, ds.mean(agg_name))
gp = orders.groupby('order_id').agg({'order_products_value': 'sum', 'customer_zip_code_prefix': 'max'})
gp = gp.groupby('customer_zip_code_prefix')['order_products_value'].mean().to_frame()
avg_ticket = brazil_geo.join(gp)
agg_name = 'avg_ticket'
avg_ticket[agg_name] = avg_ticket.order_products_value
plot_map(avg_ticket, 'Orders Average Ticket (R$)', ds.mean(agg_name), agg_name, cmap=bgy)
create_map(avg_ticket, bgy, ds.mean('avg_ticket'))
gp = orders.groupby('order_id').agg({'order_products_value': 'sum', 'order_freight_value': 'sum', 'customer_zip_code_prefix': 'max'})
agg_name = 'freight_ratio'
gp[agg_name] = gp.order_freight_value / gp.order_products_value
gp = gp.groupby('customer_zip_code_prefix')[agg_name].mean().to_frame()
freight_ratio = brazil_geo.join(gp)
plot_map(freight_ratio, 'Orders Average Freight Ratio', ds.mean(agg_name), agg_name, cmap=bgy)
create_map(freight_ratio, bgy, ds.mean('freight_ratio'))
orders['order_delivered_customer_date'] = pd.to_datetime(orders.order_delivered_customer_date)
orders['order_aproved_at'] = pd.to_datetime(orders.order_aproved_at)
orders['actual_delivery_time'] = orders.order_delivered_customer_date - orders.order_aproved_at
orders['actual_delivery_time'] = orders['actual_delivery_time'].dt.days
gp = orders.groupby('customer_zip_code_prefix')['actual_delivery_time'].mean().to_frame()
delivery_time = brazil_geo.join(gp)
agg_name = 'avg_delivery_time'
delivery_time[agg_name] = delivery_time['actual_delivery_time']
plot_map(delivery_time, 'Orders Average Delivery Time (days)', ds.mean(agg_name), agg_name, cmap=bjy)
create_map(delivery_time, bjy, ds.mean(agg_name))
pr = filter_data('state', 'pr').set_index('zip_code_prefix')
gp = orders.groupby('customer_zip_code_prefix')['actual_delivery_time'].mean().to_frame()
pr_delivery_time = pr.join(gp)
pr_delivery_time[agg_name] = pr_delivery_time['actual_delivery_time']
plot_map(pr_delivery_time, 'Orders Average Delivery Time in Parana State (days)', ds.mean(agg_name), agg_name, cmap=bjy)
create_map(pr_delivery_time, bjy, ds.mean(agg_name))
riodejaneiro = filter_data('city', 'rio de janeiro').set_index('zip_code_prefix')
gp = orders.groupby('customer_zip_code_prefix')['actual_delivery_time'].mean().to_frame()
rj_delivery_time = riodejaneiro.join(gp)
rj_delivery_time[agg_name] = rj_delivery_time['actual_delivery_time']
plot_map(rj_delivery_time, 'Orders Average Delivery Time in Rio de Janeiro (days)', ds.mean(agg_name), agg_name, cmap=bjy)
create_map(rj_delivery_time, bjy, ds.mean(agg_name))
saopaulo = filter_data('city', 'sao paulo').set_index('zip_code_prefix')
gp = orders.groupby('customer_zip_code_prefix')['actual_delivery_time'].mean().to_frame()
sp_delivery_time = saopaulo.join(gp)
sp_delivery_time[agg_name] = sp_delivery_time['actual_delivery_time']
plot_map(sp_delivery_time, 'Orders Average Delivery Time in Sao Paulo (days)', ds.mean(agg_name), agg_name, cmap=bjy)
create_map(sp_delivery_time, bjy, ds.mean(agg_name))
poa = filter_data('city', 'porto alegre').set_index('zip_code_prefix')
gp = orders.groupby('customer_zip_code_prefix')['actual_delivery_time'].mean().to_frame()
poa_delivery_time = poa.join(gp)
poa_delivery_time[agg_name] = poa_delivery_time['actual_delivery_time']
plot_map(poa_delivery_time, 'Orders Average Delivery Time in Porto Alegre (days)', ds.mean(agg_name), agg_name, cmap=bjy)
create_map(poa_delivery_time, bjy, ds.mean(agg_name))
