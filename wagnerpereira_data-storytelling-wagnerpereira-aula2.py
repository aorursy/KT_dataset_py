print('Hello World!')
'Hello World!'
texto = 'isto é um texto'

texto
inteiro = 12

inteiro
tipofloat = 12.5

tipofloat
tipofloat = inteiro

tipofloat = texto

texto
lista = [1, 'item', [1,2]]

lista

for item in lista:

    print(item)



print('saiu do for')
lista[2]
lista[-2]
idx = 0 



while idx < 10:

    print(idx)

    idx +=1
dicionario = {"id": 252,

              "preço": 35.50,

              "cor": "Azul" 

             }

dicionario
dicionario['cor']
produtos = [{"id": 252,

"preço": 30.00,

"cor": "Azul" 

},

{"id": 253,

"preço": 35.50,

"cor": "Preto" 

},

{"id": 254,

"preço": 99.90,

"cor": "Azul" 

}

]

produtos
for produto in produtos:

    print(produto['cor'])
import bq_helper



from bq_helper import BigQueryHelper



nyc = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="new_york")



query = """SELECT 

pickup_datetime,

dropoff_datetime,

passenger_count,

trip_distance * 1.6 as trip_distance,

pickup_longitude,

pickup_latitude,

dropoff_longitude,

dropoff_latitude,

payment_type,

total_amount

FROM

  `bigquery-public-data.new_york.tlc_yellow_trips_2016`

  

where pickup_longitude is not null

and dropoff_longitude is not null

and pickup_longitude > -75

and pickup_longitude < -72

and pickup_latitude < 42

and pickup_latitude > 40

and trip_distance * 1.6 < 5

and passenger_count = 1



LIMIT 100000



;

"""

data = nyc.query_to_pandas_safe(query, max_gb_scanned=20)

data.head()
data = data[data['pickup_longitude']< -60]

data.plot(x='pickup_longitude', y='pickup_latitude', kind='scatter')
import pandas as pd

from datashader.utils import lnglat_to_meters as webm

x, y = webm(data.pickup_longitude, data.pickup_latitude)

data['x'] = pd.Series(x)

data['y'] = pd.Series(y)

data.head()
x_range = (-8242000,-8210000)

y_range = (4965000, 4990000)

data = data[(data['x'] > x_range[0]) & (data['x'] < x_range[1])]

data = data[(data['y'] > y_range[0]) & (data['y'] < y_range[1])]

data.head()
# plot wtih datashader - image with black background

import datashader as ds

from datashader import transfer_functions as tf

from functools import partial

from datashader.utils import export_image

from IPython.core.display import HTML, display

from datashader.colors import colormap_select, Greys9

from colorcet import fire, rainbow, bgy, bjy, bkr, kb, kr



background = "black"

cm = partial(colormap_select, reverse=(background!='black'))

export = partial(export_image, background = background, export_path="export")

display(HTML("<style>.container { width:100% !important; }</style>"))

W = 1400 



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

    img = tf.shade(agg, cmap=cmap, how='log')

    return export(img, export_name)

create_map(data, fire, ds.count(), 'qtde_corridas')
import pandas as pd

data.head(10)
data.columns
data.describe()
data_filtrado = data[data['pickup_latitude']>0]

data_filtrado.describe()
data['passenger_count'].sum() / data['passenger_count'].count()
data.info()
data.plot(x='trip_distance', y ='total_amount', kind ='scatter')
data[(data['trip_distance'] < 50) & (data['trip_distance'] > 0)]['trip_distance'].hist()
data['pickup_datetime']


data['pickup_hourminute'] = data['pickup_datetime'].dt.strftime('%H:%M')



data['pickup_hourminute']
corridas_dia = data.groupby(by='pickup_hourminute')['trip_distance'].count()
corridas_dia.plot.line()
look_up = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May',6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}

look_up2 = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri',5: 'Sat', 6: 'Sun'}
data['pickup_month'] = data['pickup_datetime'].dt.month

data['Weekday'] = data['pickup_datetime'].dt.weekday

data['pickup_month'] = data['pickup_month'].apply(lambda x: look_up[x])

data['Weekday'] = data['Weekday'].apply(lambda x: look_up2[x])

data['pickup_day'] = data['pickup_datetime'].dt.day

data['pickup_daymonth'] = data['pickup_datetime'].dt.strftime('%d/%m')

data['Hour'] = data['pickup_datetime'].dt.strftime('%H')

data.head()

data['Weekday'] = pd.Categorical (data['Weekday'],categories = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'], ordered = True)

##data['pickup_weekday']=data['pickup_weekday'].astype(weekday_category)
data.pickup_daymonth.value_counts()
data['pickup_daymonth'].count() / data['pickup_daymonth'].nunique() 
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()



# Draw a heatmap with the numeric values in each cell

f, ax = plt.subplots(figsize=(9, 6))

sns.heatmap(flights, annot=True, fmt="d", linewidths=.5, ax=ax)
data.head()
corridas_diaSem_Hor = data.groupby (by=['Weekday','Hour'], as_index=False).count()

corridas_diaSem_Hor.head()
data.head()
corridas_diaSem_Hor = corridas_diaSem_Hor[['Weekday','Hour','pickup_datetime']]

corridas_diaSem_Hor.head()
corridas_diaSem_Hor.columns = ['Weekday','Hour','qty']

corridas_diaSem_Hor.head()
corridas_diaSem_Hor_pivot = pd.pivot_table(data=corridas_diaSem_Hor,

                    index='Weekday',

                    values='qty',

                    columns='Hour')

corridas_diaSem_Hor_pivot
colors = ["lightgrey","lightgrey","#FCBCAF","#FCBCAF","#FCBCAF","red", "darkred"]

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



f, ax = plt.subplots(figsize=(15, 5))

sns.heatmap(corridas_diaSem_Hor_pivot, annot=False, linewidths=.5, cmap=colors)
corridas_diaSem_Hor_pivot = pd.pivot_table(data=corridas_diaSem_Hor,

                    index='Hour',

                    values='qty',

                    columns='Weekday')

corridas_diaSem_Hor_pivot
import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



f, ax = plt.subplots(figsize=(20, 10))

sns.heatmap(corridas_diaSem_Hor_pivot, annot=False, linewidths=.5, cmap="YlGnBu")
import matplotlib.pyplot as plt

import numpy as np

import matplotlib.colors



norm = matplotlib.colors.Normalize(0,1)

colors = [[norm(0.3), "white"],

          [norm(0.4), "lightgrey"],

          [norm(0.6), "lightgrey"],

          [norm(0.7), "red"]]



cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)