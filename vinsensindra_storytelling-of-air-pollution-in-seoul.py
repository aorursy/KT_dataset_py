import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from bokeh.io import output_file,show,output_notebook,push_notebook

from bokeh.plotting import figure

from bokeh.models import BoxAnnotation

from bokeh.models.widgets import Tabs,Panel

from bokeh.models.formatters import DatetimeTickFormatter

output_notebook()
df = pd.read_csv('../input/air-pollution-in-seoul/AirPollutionSeoul/Measurement_summary.csv')

df_item = pd.read_csv('../input/air-pollution-in-seoul/AirPollutionSeoul/Original Data/Measurement_item_info.csv')

df_measure = pd.read_csv('../input/air-pollution-in-seoul/AirPollutionSeoul/Original Data/Measurement_info.csv')

df_station = pd.read_csv('../input/air-pollution-in-seoul/AirPollutionSeoul/Original Data/Measurement_station_info.csv')



# Change measurement date to datetime type, and separate them

# Set indexing on station code, date, and time

df['Measurement date'] = pd.to_datetime(df['Measurement date'])

df['date'] = [d.date() for d in df['Measurement date']]

df['time'] = [d.time() for d in df['Measurement date']]



df_measure['Measurement date'] = pd.to_datetime(df_measure['Measurement date'])



item_codes = {

    1:'SO2', 

    3:'NO2', 

    5:'CO',

    6:'O3',

    8:'PM10',

    9:'PM2.5'

}



# Mapping item code

df_measure['pollutant'] = df_measure['Item code'].apply(lambda x: item_codes[x]) 
pollutant_cols = ['SO2', 'NO2', 'CO', 'O3', 'PM10', 'PM2.5']

def filter_normal(df_1, df_2, col):

    return df_2[df_2['pollutant'] == col].merge(df_1[['Measurement date', 'Station code', col]], on=['Measurement date', 'Station code'])



df_pollutant = {}

for c in pollutant_cols:

    df_merged = filter_normal(df, df_measure[df_measure['Instrument status'] == 0], c)

    df_merged['date'] = df_merged['Measurement date'].dt.date

    df_merged['time'] = df_merged['Measurement date'].dt.time

    df_pollutant[c] = df_merged.copy()
# Tabs of mean in each station

units = {

    'SO2':'ppm', 

    'NO2':'ppm', 

    'CO':'ppm',

    'O3':'ppm',

    'PM10':'Mircrogram/m3',

    'PM2.5':'Mircrogram/m3'

}



lp = []

for c in pollutant_cols:

    p = figure(plot_width=700, plot_height=400)

    dt = df_pollutant[c].groupby(['Station code', 'time']).mean().reset_index()

    for x in df_station['Station code'].unique():

        dtemp = dt[dt['Station code'] == x]

        p.line(dtemp['time'], dtemp['Average value'], line_width=1)

    

    label = df_item[df_item['Item name'] == c]

    box = BoxAnnotation(top=float(label['Good(Blue)']), fill_alpha=0.1, fill_color='gray')

    p.add_layout(box)

    box = BoxAnnotation(bottom=float(label['Good(Blue)']), top=float(label['Normal(Green)']), fill_alpha=0.1, fill_color='blue')

    p.add_layout(box)

    box = BoxAnnotation(bottom=float(label['Normal(Green)']), top=float(label['Bad(Yellow)']), fill_alpha=0.1, fill_color='green')

    p.add_layout(box)

    box = BoxAnnotation(bottom=float(label['Bad(Yellow)']), top=float(label['Very bad(Red)']), fill_alpha=0.1, fill_color='yellow')

    p.add_layout(box)

    box = BoxAnnotation(bottom=float(label['Very bad(Red)']), fill_alpha=0.1, fill_color='red')

    p.add_layout(box)



    p.xaxis.formatter = DatetimeTickFormatter(hours='%Hh')

    p.xaxis.axis_label = 'Time (h)'

    p.yaxis.axis_label = units[c]



    tab = Panel(child=p, title=c)

    lp.append(tab)

    

tabs = Tabs(tabs=lp)

show(tabs)
# Tabs of overall means

lp = []

for c in pollutant_cols:

    p = figure(plot_width=700, plot_height=400, x_axis_type='datetime')

    dt = df_pollutant[c].groupby(['time']).mean().reset_index()

    p.line(dt['time'], dt['Average value'], line_width=1)

    

    label = df_item[df_item['Item name'] == c]

    box = BoxAnnotation(top=float(label['Good(Blue)']), fill_alpha=0.1, fill_color='gray')

    p.add_layout(box)

    box = BoxAnnotation(bottom=float(label['Good(Blue)']), top=float(label['Normal(Green)']), fill_alpha=0.1, fill_color='blue')

    p.add_layout(box)

    box = BoxAnnotation(bottom=float(label['Normal(Green)']), top=float(label['Bad(Yellow)']), fill_alpha=0.1, fill_color='green')

    p.add_layout(box)

    box = BoxAnnotation(bottom=float(label['Bad(Yellow)']), top=float(label['Very bad(Red)']), fill_alpha=0.1, fill_color='yellow')

    p.add_layout(box)

    box = BoxAnnotation(bottom=float(label['Very bad(Red)']), fill_alpha=0.1, fill_color='red')

    p.add_layout(box)

    

    p.xaxis.formatter = DatetimeTickFormatter(hours='%Hh')

    p.xaxis.axis_label = 'Time (h)'

    p.yaxis.axis_label = units[c]

    

    tab = Panel(child=p, title=c)

    lp.append(tab)

    

tabs = Tabs(tabs=lp)

show(tabs)