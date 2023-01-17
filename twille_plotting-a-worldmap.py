import pandas as pd

import geopandas as gpd



df_series = pd.read_csv("../input/world-development-indicators/Series.csv")

df_indicators = pd.read_csv("../input/world-development-indicators/Indicators.csv")



df_series.head()
df_series = df_series.filter(items=["SeriesCode", "Topic", "IndicatorName"])

df_series.Topic.unique()
df_series[df_series.Topic == "Health: Population: Dynamics"]
df_indicators.head()
df_pop = df_indicators[df_indicators.IndicatorCode == 'SP.POP.GROW']

df_pop.Year.unique()
pop_2014 = df_pop[df_pop.Year == 2014]

pop_2014 = pop_2014.filter(["CountryCode", "Value"])

pop_2014.rename(columns={"Value": "Population growth"}, inplace=True)



world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

world.head()
world = pd.merge(world, pop_2014, left_on="iso_a3", right_on="CountryCode")

world = world.sort_values(by=["Population growth"], ascending=False)

world.head(5)
world.plot(column="Population growth", legend=True, figsize=(24, 12), cmap="Reds")