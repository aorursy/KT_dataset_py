import numpy as np

import pandas as pd



from geopy.distance import geodesic
df_address = pd.read_csv('../input/testing-google-maps-geocoding-api/address_coordinates.csv')

df_stn = pd.read_csv('../input/singapore-train-station-coordinates/mrt_lrt_data.csv')
df_address.head()
df_address.info()
df_stn.head()
def compute_distance(address_row, df_stn):

    address_lat, address_lng = address_row[['lat','lng']]

    min_distance = 9999.0

    for stn_lat, stn_lng in zip(df_stn['lat'], df_stn['lng']):

        distance = geodesic((address_lat, address_lng), (stn_lat, stn_lng)).km

        if distance < min_distance:

            min_distance = distance

    return min_distance
distance_series = df_address.apply(compute_distance, axis=1, df_stn=df_stn)
df_output = pd.DataFrame({'address': df_address['address'], 'distance': distance_series})

df_output.to_csv('address_to_nearest_stn_dist.csv', index=False)