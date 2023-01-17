import pandas as pd
import plotly
import plotly.express as px
from pycountry_convert import country_name_to_country_alpha3
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv('../input/population-by-country-2020/population_by_country_2020.csv')
data
def get_iso(col):
    try:
        iso_3 =  country_name_to_country_alpha3(col)
    except:
        iso_3 = 'Unknown'
    return iso_3

data['iso_alpha'] = data['Country (or dependency)'].apply(lambda x: get_iso(x))
import warnings
warnings.filterwarnings('ignore')
data['iso_alpha'].loc[data['Country (or dependency)'] == "DR Congo"] = 'COD'
data['iso_alpha'].loc[data['Country (or dependency)'] == "Czech Republic (Czechia)"] = 'CZE'
data['iso_alpha'].loc[data['Country (or dependency)'] == "State of Palestine"] = 'PSE'
fig = px.choropleth(data, locations="iso_alpha",
                    color="Population (2020)",
                    hover_name="Country (or dependency)",
                    color_continuous_scale='Viridis_r')
fig.update_layout(title_text="World population in 2020")
fig.show()
fig = px.choropleth(data[data['Density (P/Km²)'] < 600], locations="iso_alpha",
                    color="Density (P/Km²)",
                    hover_name="Country (or dependency)",
                    color_continuous_scale='Reds')
fig.update_layout(title_text="Population Density (P/Km²)")
fig.show()
data['change_yearly'] = [float(x.split(' ')[0]) for x in data['Yearly Change']]
fig = px.choropleth(data, locations="iso_alpha",
                    color=data["change_yearly"],
                    hover_name="Country (or dependency)",
                    color_continuous_scale=px.colors.sequential.Plasma)
fig.update_coloraxes(colorbar_title="Population change (%)")
fig.update_layout(title_text="World Population change (in percentage)")
fig.show()
fig = px.choropleth(data, locations="iso_alpha",
                    color="Migrants (net)",
                    hover_name="Country (or dependency)",
                    color_continuous_scale=px.colors.sequential.Plasma)
fig.update_layout(title_text="Migrants")
fig.show()
data_f_no_na = data[data["Fert. Rate"] != 'N.A.']
fig = px.choropleth(data_f_no_na, locations="iso_alpha",
                    color=data_f_no_na["Fert. Rate"].astype('float'),
                    hover_name="Country (or dependency)")
fig.update_coloraxes(colorbar_title="Fertility Rate",colorscale="Greens")
fig.update_layout(title_text="Fertility Rate")
fig.show()
data_u_no_na = data[data["Urban Pop %"] != 'N.A.']
data_u_no_na['Pop_urban'] = [int(x.split(' ')[0]) for x in data_u_no_na['Urban Pop %']]
fig = px.choropleth(data_u_no_na, locations="iso_alpha",
                    color=data_u_no_na["Pop_urban"],
                    hover_name="Country (or dependency)")
fig.update_coloraxes(colorbar_title="Urban population (%)",colorscale="Cividis_r")
fig.update_layout(title_text="Urban population (in percentage)")
fig.show()
