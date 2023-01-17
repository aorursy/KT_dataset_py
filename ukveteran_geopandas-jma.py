import geopandas
df = geopandas.read_file(geopandas.datasets.get_path('nybb'))

ax = df.plot(figsize=(10, 10), alpha=0.5, edgecolor='k')
import geopandas

import geoplot



world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))

cities = geopandas.read_file(geopandas.datasets.get_path('naturalearth_cities'))
world.head()
world.plot()
world = world[(world.pop_est>0) & (world.name!="Antarctica")]

world['gdp_per_cap'] = world.gdp_md_est / world.pop_est

world.plot(column='gdp_per_cap');
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1)

world.plot(column='gdp_per_cap', ax=ax, legend=True)
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1)

world.plot(column='pop_est', ax=ax, legend=True)