!pip install pycountry
import pandas as pd

df = pd.read_csv("../input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv")
country_col = "Country"

state_col = "Province/State"

target = "Confirmed"
import pycountry

countries = {}

for country in pycountry.countries:

    countries[country.name] = country.alpha_3

df["iso_alpha"] = df[country_col].map(countries.get)
df.head()
data = df.groupby("iso_alpha")[target].sum().reset_index()
import plotly.express as px

# df2 = px.data.gapminder().query("year == 2007")

fig = px.choropleth(data, locations="iso_alpha",

                     color=target, # which column to use to set the color of markers

#                      hover_name=country_col, # column added to hover information

                     projection="natural earth")

fig.show()