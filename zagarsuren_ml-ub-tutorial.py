import numpy as np

import pandas as pd

import datetime as dt

import folium

pd.set_option('display.max_rows', 50)
def read_data(train_path, test_path, routes_path, stops_path):

    # read_data

    train_tmp = pd.read_csv(train_path)

    test = pd.read_csv(test_path)

    routes = pd.read_csv(routes_path)

    stops = pd.read_csv(stops_path)

    # merge

    train = train_tmp.merge(stops, on = 'BUSSTOP_ID', how = 'left')

    return train, test, routes, stops
def generate_features(df):

    # RECORD_DATE - г datetime формат руу хөрвүүлэх

    df['RECORD_DATE'] = pd.to_datetime(df['RECORD_DATE'])

    df['ymd'] = df['RECORD_DATE'].dt.strftime('%Y-%m-%d')

    # Буудал хоорондох хугацааг тооцох

    df = df.sort_values(by = ['BUS_ID','TIMESTAMP'], ascending = ['False', 'False'])

    df['time_spent'] = df.groupby(['BUS_ID', 'ymd'])['TIMESTAMP'].diff()

    df['busstop_lag'] = df.groupby(['BUS_ID', 'ymd'])['BUSSTOP_NMMN'].shift(1)

    df['busstop_from_to'] = df["busstop_lag"] + " >> " + df["BUSSTOP_NMMN"]

    del df['busstop_lag']

    return df
train_path = '/kaggle/input/data-nomads-phase-1-competition/training.csv'

test_path = '/kaggle/input/data-nomads-phase-1-competition/test.csv'

routes_path = '/kaggle/input/data-nomads-phase-1-competition/routes.csv'

stops_path = '/kaggle/input/data-nomads-phase-1-competition/stops.csv'
train, test, routes, stops = read_data(train_path, test_path, routes_path, stops_path)
print('Автобусны тоо::', len(train['BUS_ID'].unique()))

print('Чиглэлийн тоо::', len(train['BUSROUTE_ID'].unique()))

print('Автобусны буудлын тоо::', len(train['BUSSTOP_ID'].unique()))
# Уртраг өргөрөгийн мэдээллийг ашиглан автобусны буудлуудын мэдээллийг газрын зурагт харъя.

latlon = [47.91053, 106.90698]

m = folium.Map(location=latlon,

               zoom_start=13)



tiles = 'https://api.mapbox.com/v4/mapbox.light/{z}/{x}/{y}.png?access_token=pk.eyJ1IjoidWd1dWRlaSIsImEiOiJjanYyMHQ1aDkwbnRiNGVwOTZ0NnVodzB2In0.sv3Wodc2Kb_BIyhNWQDVqg'

folium.TileLayer(name='Mapbox',

                 tiles=tiles,

                 overlay=True,

                 attr='by Mapbox').add_to(m)



bus_stops = folium.map.FeatureGroup()



for lat, lng, name in zip(stops.GPS_COORDY, stops.GPS_COORDX, stops.BUSSTOP_NMMN):

    bus_stops.add_child(

         folium.vector_layers.CircleMarker(

            [lat, lng],

            popup=name,

            radius = 3,

            color='green',

            fill=True,

            fill_color='green'

        )

    )

m.add_child(bus_stops)
train = generate_features(train)
train[:20]
# Буудал хооронд явсан дундаж хугацаа

tmp = pd.DataFrame(train.groupby(['BUS_ID','busstop_from_to'])['time_spent'].mean().reset_index()).dropna()

tmp
tmp['time_spent'].describe()