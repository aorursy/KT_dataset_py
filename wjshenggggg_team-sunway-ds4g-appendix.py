!pip install rasterstats --quiet
import os

import glob 

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from datetime import datetime



import rasterio as rio

import folium

import tifffile



# Geospatial libraries that we will be using for this

import rasterio

import rasterstats

import folium

import geopandas as gpd

from shapely.geometry import Point



from tqdm.notebook import trange, tqdm



import warnings

warnings.simplefilter(action='ignore')



pd.options.display.max_columns = None
DATA_DIR = '../input/ds4g-environmental-insights-explorer/eie_data/'

os.listdir(DATA_DIR)
eia_df_2018 = pd.read_excel('../input/eia-data/eia_puerto_rico_2018.xlsx')

eia_df_2017 = pd.read_excel('../input/eia-data/eia_puerto_rico_2017.xlsx')
manual_map = {'AES ILUMINA': 'A.E.S. Corp.',

                'AES Puerto Rico': 'AES Ilumina',

                'Aguirre Plant': 'Aguirre',

                'Cambalache Plant': 'Cambalache',

                'Caonillas': 'Caonillas 1',

                'Central San Juan Plant': 'San Juan CC', #'Caonillas 2'

                'Costa Sur Plant': 'Costa Sur', # Carite 1

                # 'Culebra': ''

                'Daguao': 'Daguao',

                'EcoElectrica': 'EcoEléctrica', # 'Dos Bocas'

                # 'HIMA San Pablo - Caguas': ''

                # 'Holsum de Puerto Rico, Inc.': 'Garzas 1'

                # 'Humacao Solar Project, LLC': 'Garzas 2'

                # 'Janssen Ortho LLC': 'Jobos'

                'Jobos': 'Jobos', # Loiza Solar Park

                'Mayaguez Plant': 'Mayagüez',

                'Oriana Energy Hybrid': 'Oriana Solar Farm',

                'Palo Seco Plant': 'Palo Seco',

                'Pattern Santa Isabel LLC': 'Santa Isabel Wind Farm',

                'Punta Lima Wind Farm': 'Punta Lima', # 'Río Blanco'

                'San Fermin Solar Farm': 'San Fermin Solar Farm', # 'Salinas'

                'Vega Baja': 'Vega Baja',

                'Vieques': 'Vieques EPP', # 'San Juan CC'

                'Yabucoa': 'Yabucoa'}
# https://www.kaggle.com/maxlenormand/saving-the-power-plants-csv-to-geojson

power_plants_path = '../input/gppd-geojson/Geolocated_gppd_120_pr.geojson'

power_plants_original = gpd.read_file(power_plants_path)



# we removed plants that do not exist in EIA data

power_plants_path = '../input/geojson-matched/Geolocated_gppd_120_pr_matched.geojson'

power_plants = gpd.read_file(power_plants_path)
# eia_data = pd.read_excel('../input/eiadata/eia_puerto_rico_2018.xlsx')



eia_df_2018 = pd.read_excel('../input/eia-data/eia_puerto_rico_2018.xlsx')

eia_df_2017 = pd.read_excel('../input/eia-data/eia_puerto_rico_2017.xlsx')



def process_eia_df(eia_data, power_plants):

    eia_data.columns = eia_data.columns.str.replace('\n', ' ')

    

    eia_data['mapped_name'] = eia_data['Plant Name'].map(manual_map)



    eia_df = pd.merge(eia_data, power_plants, left_on='mapped_name', right_on='name', how='inner')



    return eia_df



def calc_netgen(eia_df):

    netgen_months = [col for col in eia_df.columns if col.find('Netgen') != -1]



    netgen_months.extend(['system:index', 'primary_fuel'])



    tmp = eia_df[netgen_months]



    tmp = tmp.replace('.', 0)



    netgen_sum_month = tmp.sum(axis=0)[:-2].values

    

    return netgen_sum_month

    

def plot_netgen(eia_df_2017, eia_df_2018):

    netgen_sum_month = []

    netgen_sum_month.extend(calc_netgen(eia_df_2017))

    netgen_sum_month.extend(calc_netgen(eia_df_2018))

    

    time = [str(i)+'-17' for i in range(1, 13)]

    time.extend([str(i)+'-18' for i in range(1, 13)])

    

    # plt.figure(figsize=(8, 6))

    plt.xticks(rotation=45)

    plt.title('Total electricity generated from 2017 to 2018')

    plt.xlabel('Time')

    plt.ylabel('Electricity generated')

    plt.plot(time, netgen_sum_month)

    plt.show()
eia_df_2018 = process_eia_df(eia_df_2018, power_plants)

eia_df_2017 = process_eia_df(eia_df_2017, power_plants)



plot_netgen(eia_df_2017, eia_df_2018)
def filter_nan(eia_df):

    df_sub = eia_df[eia_df['mapped_name'].isna()]

    df_sub_filled = eia_df[~eia_df['mapped_name'].isna()]

    return df_sub, df_sub_filled



df_sub, df_sub_filled = filter_nan(eia_df_2018)



df_sub_filled.head()
# https://disc.gsfc.nasa.gov/datasets/OMNO2_003/summary?keywords=OMI

omi = pd.read_csv("../input/omiv1/OMI-Aura_L2-OMNO2.csv")



omi.head()
def scale_lat_to_img_idx(lat):

    lat = float(lat)

    lat_img_idx = (18.6 - lat) * 148 / (18.6 - 17.9)

    return int(lat_img_idx)



def scale_lon_to_img_idx(lon):

    lon = float(lon)

    lon_img_idx = (67.3 + lon) * 475 / (67.3 - 65.2)

    return int(lon_img_idx)
omi['img_idx_lt'] = omi['Latitude'].apply(scale_lat_to_img_idx)



omi['img_idx_lg'] = omi['Longitude'].apply(scale_lon_to_img_idx)



print('len of omi before filter:', len(omi))



# filter out of bounds

omi = omi[(omi['img_idx_lt'] >= 0) & (omi['img_idx_lt'] <= 148)]

omi = omi[(omi['img_idx_lg'] >= 0) & (omi['img_idx_lt'] <= 475)]



omi.reset_index(inplace=True, drop=True)



print('len of omi after filter:', len(omi))
power_plants['img_idx_lt'] = power_plants['latitude'].apply(scale_lat_to_img_idx)



power_plants['img_idx_lg'] = power_plants['longitude'].apply(scale_lon_to_img_idx)
non_rewew_energy = ['Oil', 'Coal', 'Gas']



non_renew_energy_plants = power_plants[power_plants['primary_fuel'].isin(non_rewew_energy)]

renew_energy_plants = power_plants[~power_plants['primary_fuel'].isin(non_rewew_energy)]



non_renew_energy_plants.head()



len(non_renew_energy_plants), len(renew_energy_plants)
# Plot on the map

lat = 18.200178; lon = -66.664513



def plot_polygons_on_Map(geodataframe,

                         original_latitude=lat,

                         original_longitude=lon,

                         zoom=9):

    plot = folium.Map(location = (original_latitude, original_longitude), zoom_start=zoom)



    geojson = folium.GeoJson(geodataframe,

                             style_function=lambda x: {

                                 'Color':'white',

                                 'fillColor':'red' if x['properties']['primary_fuel'] in ['Coal', 'Oil', 'Gas']

                                                   else 'white'})

    popup = folium.Popup(str(geodataframe.primary_fuel))

    popup.add_to(geojson)

    geojson.add_to(plot)



    return(plot)
buffered_power_plants = power_plants.copy()



buffered_power_plants['geometry'] = power_plants.geometry.buffer(0.1)



plot_polygons_on_Map(buffered_power_plants)
polluted_idx = non_renew_energy_plants[['img_idx_lt', 'img_idx_lg']].values



non_polluted_idx = renew_energy_plants[['img_idx_lt', 'img_idx_lg']].values



examples = omi[['img_idx_lt', 'img_idx_lg']].values



DEFINED_BOUNDS = 15



def in_polluted_bounds(polluted_idx, lat, lon):

    for pol_lat, pol_lon in polluted_idx:

        lat_in_bounds = pol_lat - DEFINED_BOUNDS <= lat <= pol_lat + DEFINED_BOUNDS

        lon_in_bounds = pol_lon - DEFINED_BOUNDS <= lon <= pol_lon + DEFINED_BOUNDS

        if lat_in_bounds and lon_in_bounds:

            return True

    return False
# omi['is_polluted_bounds'] = omi.apply(lambda x: in_polluted_bounds(polluted_idx, x['img_idx_lat'], x['img_idx_lon']), axis=1)



polluted_bounds_bool = [0] * len(omi)



for idx, row_data in omi.iterrows():

    lat, lon = row_data['img_idx_lt'], row_data['img_idx_lg']

    if in_polluted_bounds(polluted_idx, lat, lon):

        polluted_bounds_bool[idx] = 1



omi['is_polluted_bounds'] = polluted_bounds_bool



non_polluted_bounds_bool = [0] * len(omi)



for idx, row_data in omi.iterrows():

    lat, lon = row_data['img_idx_lt'], row_data['img_idx_lg']

    if in_polluted_bounds(non_polluted_idx, lat, lon):

        non_polluted_bounds_bool[idx] = 1



omi['not_polluted_bounds'] = non_polluted_bounds_bool
# omi.dropna(inplace=True)



omi['is_polluted_bounds'].dropna(inplace=True)

omi['not_polluted_bounds'].dropna(inplace=True)
omi.head()
from scipy import stats



def run_t_test(population_1, population_2, num_tests=10, num_samples=500):

    count = 0



    for i in range(num_tests):

        np.random.seed(i)        



        sample_polluted = np.random.choice(population_1, num_samples)

        sample_non_polluted = np.random.choice(population_2, num_samples)



        _, p_value = stats.ttest_ind(sample_polluted, sample_non_polluted)



        if p_value < 0.05:

            count += 1

            

    return count / num_tests
col = 'ColumnAmountNO2'



dirty_no2 = omi[omi['is_polluted_bounds'] == 1][col].mean() 

clean_no2 = omi[omi['is_polluted_bounds'] == 0][col].mean()



polluted_higher = (dirty_no2 - clean_no2) / clean_no2

pct_stats_signific = run_t_test(omi[omi['is_polluted_bounds'] == 1][col], omi[omi['is_polluted_bounds'] == 0][col])



polluted_higher, pct_stats_signific
dirty_no2 = omi[omi['is_polluted_bounds'] == 1][col].mean() 

clean_no2 = omi[omi['not_polluted_bounds'] == 1][col].mean()



polluted_higher = (dirty_no2 - clean_no2) / clean_no2

pct_stats_signific = run_t_test(omi[omi['is_polluted_bounds'] == 1][col], omi[omi['not_polluted_bounds'] == 1][col])



polluted_higher, pct_stats_signific
df_s5 = pd.DataFrame()



files = []

captured_datetime = []



for dirname, _, filenames in os.walk(os.path.join(DATA_DIR, 's5p_no2')):

    for filename in filenames:

        files.append(os.path.join(dirname, filename))

        captured_datetime.append(datetime.strptime(filename.split('_')[-2],'%Y%m%dT%H%M%S').date())

        

df_s5['path'] = files

df_s5['captured_date'] = captured_datetime



# Sort dataframe by ascending time

df_s5.sort_values('captured_date', inplace=True)

df_s5.reset_index(inplace=True, drop=True)



df_s5.head()
# https://www.kaggle.com/maxlenormand/saving-the-power-plants-csv-to-geojson

power_plants_path
def create_df(df_s5, power_plants_path, df, BAND=1):

    for idx, row_data in tqdm(df_s5.iterrows()):

        raster_stats = rasterstats.zonal_stats(power_plants_path,

                                               row_data['path'],

                                               band=BAND,

                                               stats=['mean'])



        raster_stats_list = [plant['mean'] for plant in raster_stats]



        df[row_data['captured_date']] = raster_stats_list

        

    return df



# N02_measurments_df = pd.DataFrame({'index_power_plant': power_plants['system:index']})

# N02_measurments_df = create_df(df_s5, power_plants_path, N02_measurments_df, 1)
features = ['NO2_column_number_density',

            'tropospheric_NO2_column_number_density',

            'stratospheric_NO2_column_number_density',

            'NO2_slant_column_number_density',

            'tropopause_pressure',

            'absorbing_aerosol_index',

            'cloud_fraction',

           ]



tmp = pd.DataFrame({'index_power_plant': power_plants['system:index']})



for idx, feat in enumerate(features):

    df = create_df(df_s5, power_plants_path, tmp, idx+1)

    

    filename = feat + '.csv'

    

    df.to_csv(filename, index=False)
def calc_avg_month(df, feature):

    power_plants = df.iloc[:, 0]

    tmp = df.iloc[:, 1:]

    tmp.columns = pd.to_datetime(tmp.columns)

    

    # transpose & reset

    tmp = tmp.T

    tmp.reset_index(inplace=True)

    

    # extract year

    tmp['year'] = tmp['index'].dt.year

    tmp['month'] = tmp['index'].dt.month

    

    df = pd.DataFrame()

    

    dict_name = {

        'min': feature + '_min',

        'max': feature + '_max',

        'mean': feature + '_mean',

        'std': feature + '_std'

    }        

            

    for idx, plant in enumerate(power_plants):

        subset = tmp.groupby(['year', 'month'])[idx].agg(['min', 'max', 'mean', 'std'])

        subset = subset.reset_index()

        subset['system:index'] = plant   

        subset = subset.rename(columns=dict_name)

        df = pd.concat([df, subset], axis=0, ignore_index=True)

            

    return df
def read_and_merge():

    raster_files_dir = '../input/rasterstats/rasterstats'



    raster_files = os.listdir(raster_files_dir)

    

    comb_df = pd.DataFrame()

    

    for file in raster_files:

        df = pd.read_csv(os.path.join(raster_files_dir, file))

        

        feature, _ = file.split('.')

        

        df = calc_avg_month(df, feature)

        

        comb_df = pd.concat([comb_df, df], axis=1)

            

    return comb_df.T.drop_duplicates().T
df_ras = read_and_merge()
cols = df_ras.columns



new_cols = ['system:index']



new_cols.extend([c for c in cols if c != 'system:index'])



df_ras = df_ras[new_cols]



df_ras.head()
df_ras['generated'] = np.nan



df_ras.head()
def safe_convert(x):

    try:

        x = float(x)

        return x

    except:

        return np.nan



    

df_sub, df_sub_filled = filter_nan(eia_df_2018)



netgen_months = [col for col in df_sub_filled.columns if col.find('Netgen') != -1]



tmp = df_ras.copy()



final = pd.DataFrame()



for idx, m in enumerate(netgen_months[6:], 6):

    df_sub_filled[m] = df_sub_filled[m].apply(safe_convert)

    

    g = df_sub_filled.groupby(['system:index'])[m].sum()

    

    g = g.reset_index()

    

    for _, row in g.iterrows():

        name_con = tmp['system:index'] == row['system:index']

        month_con = tmp['month'] == idx + 1

        tmp.loc[(name_con) & (month_con), 'generated'] = row[m]

    

tmp.head()
df_sub, df_sub_filled = filter_nan(eia_df_2017)



netgen_months = [col for col in df_sub_filled.columns if col.find('Netgen') != -1]



for idx, m in enumerate(netgen_months[:6]):

    df_sub_filled[m] = df_sub_filled[m].apply(safe_convert)

    

    g = df_sub_filled.groupby(['system:index'])[m].sum()

    

    g = g.reset_index()

    

    for _, row in g.iterrows():

        name_con = tmp['system:index'] == row['system:index']

        month_con = tmp['month'] == idx + 1

        tmp.loc[(name_con) & (month_con), 'generated'] = row[m]

        

tmp.tail()
tmp.to_csv('df_ras_gen.csv', index=False)