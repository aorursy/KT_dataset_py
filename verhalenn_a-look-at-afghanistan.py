import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap



data = pd.read_csv('../input/globalterrorismdb_0617dist.csv',  encoding='ISO-8859-1', low_memory=False)

data.head()
data = data[data.country_txt == 'Afghanistan']

data.shape
year_attacks = data.iyear.value_counts().sort_index()

plt.plot(year_attacks.index, year_attacks)

plt.xlabel('Year')

plt.ylabel('Number of Attacks')
# Finding the number of attacks made by each group.

data.gname.value_counts().head()
# Splitting only taliban data from the afghanistan attacks

taliban = data[data.gname == 'Taliban']
taliban.iyear.value_counts().sort_index().plot()

plt.xlabel('Year')

plt.ylabel('Number of Attacks')
taliban.weaptype1_txt.value_counts()
taliban_chemical = taliban[taliban['weaptype1_txt'] == 'Chemical']

taliban_chemical.iyear.value_counts().sort_index()
taliban['attacktype1_txt'].value_counts()
sns.jointplot(x='longitude', y='latitude', data=taliban, color='red')
# Setting up the basmap data.

m = Basemap(projection='stere', llcrnrlon=taliban.longitude.min() - 2, llcrnrlat=taliban.latitude.min() - 2,

           urcrnrlon=taliban.longitude.max() + 2, urcrnrlat=taliban.latitude.max() + 2,

           lon_0=taliban.longitude.mean(), lat_0=taliban.latitude.mean())

# Drawing countm.drawcountries()

x, y = m(taliban.longitude.values, taliban.latitude.values)

m.scatter(x, y, marker='.', color='r')

plt.show()
m.drawcountries()

pre_ten_taliban = taliban[taliban.iyear < 2010]

x, y = m(pre_ten_taliban.longitude.values, pre_ten_taliban.latitude.values)

m.scatter(x, y, marker='.', color='r')

plt.title('Taliban Attacks before 2010')

plt.show()
m.drawcountries()

post_ten_taliban = taliban[taliban.iyear >= 2010]

x, y = m(post_ten_taliban.longitude.values, post_ten_taliban.latitude.values)

m.scatter(x, y, marker='.', color='r')

plt.title('Taliban Attacks after 2010')

plt.show()
from sklearn.cluster import KMeans



# Creating and fitting a kmeans cluster to taliban coordinates data.

y_pred = KMeans(n_clusters=6).fit_predict(taliban[['latitude', 'longitude']].dropna(axis=0))
taliban_position = taliban[['latitude', 'longitude']].dropna(axis=0)

# Loop through each cluster

for i in np.unique(y_pred):

    # Seperate out the cluster from the rest of the data.

    plot_data = taliban_position[y_pred == i]

    # Plot the seperated cluster

    plt.plot(plot_data.longitude, plot_data.latitude, '.')



plt.title('Taliban Attacks Split Into 5 Clusters')
# Combining the year and month into a single value for linear regression

taliban['year_month'] = taliban['iyear'] + (taliban['imonth'] / 12)

taliban_cluster = pd.merge(taliban[['year_month']], taliban_position, how='inner', left_index=True, right_index=True)
from sklearn.linear_model import LinearRegression



colors = 'bgrmcy'

# Loop through each cluster

for i in np.unique(y_pred):

    # Create a linear model for latitude adn longitude

    lon_lm = LinearRegression()

    lat_lm = LinearRegression()

    # Seperate out the cluster form the rest of the date.

    plot_data = taliban_cluster[y_pred == i]

    # Fit the cluster data with the linear models

    lon_lm.fit(plot_data[['year_month']].values.reshape(-1, 1), plot_data['longitude'])

    lat_lm.fit(plot_data[['year_month']].values.reshape(-1, 1), plot_data['latitude'])

    # Create the prediction lines from 2010 to 2050

    time_range = np.linspace(2010, 2050, 50)

    lon_pred = lon_lm.predict(time_range.reshape(-1, 1))

    lat_pred = lat_lm.predict(time_range.reshape(-1, 1))

    # Find the prediction points for 2010

    max_lon_pred = lon_lm.predict(2010)

    max_lat_pred = lat_lm.predict(2010)

    # Plot the 2010 prediction point

    plt.plot(max_lon_pred, max_lat_pred, '.', markersize=20, color='k')

    # Plot the prediction line

    plt.plot(lon_pred, lat_pred, '--k')

    # Plot the cluster

    plt.plot(plot_data['longitude'], plot_data['latitude'], '.', alpha=0.2, color=colors[i], label=i)

    

plt.title('Movement of Attacks by Cluster in Afghanistan')

plt.legend()

plt.text(69, 34.53, 'Kabul', bbox=dict(color='yellow', alpha=0.9))
colors = 'bgrmcy'

# Loop through each cluster

for i in np.unique(y_pred):

    # Seperate the cluster data

    plot_data = taliban_cluster[y_pred == i]

    # For the left graph show longitude to time

    # This is to show a left right relationship

    plt.subplot(121)

    plt.plot(plot_data['longitude'], plot_data['year_month'], '.', color=colors[i])

    title_lon = 'Cluster ' + str(i) + ': Longitude'

    plt.title(title_lon)

    # For the right graph show time to latitude

    plt.subplot(122)

    plt.plot(plot_data['year_month'], plot_data['latitude'], '.', color=colors[i])

    title_lat = 'Cluster ' + str(i) + ': Latitude'

    plt.title(title_lat)

    plt.show()
# Loop through each cluster

colors = 'bgrmcy'



for i in np.unique(y_pred):

    taliban_sub = taliban_cluster[y_pred == i]

    # Make four clusters for each cluster

    y_sub_pred = KMeans(n_clusters=4).fit_predict(taliban_sub[['latitude', 'longitude']].dropna(axis=0))

    # Loop through each sub cluster

    for sub in np.unique(y_sub_pred):

        sub_data = taliban_sub[y_sub_pred == sub]

        # Create a Model

        lat_lm = LinearRegression()

        lon_lm = LinearRegression()

        lat_lm.fit(sub_data[['year_month']], sub_data['latitude'])

        lon_lm.fit(sub_data[['year_month']], sub_data['longitude'])

        

        # Calculate the line from 2010 to 2050

        time_range = np.linspace(2010, 2050, 50)

        lon_pred = lon_lm.predict(time_range.reshape(-1, 1))

        lat_pred = lat_lm.predict(time_range.reshape(-1, 1))

        # Calculate the point at 2010

        lon_ten = lon_lm.predict(2010)

        lat_ten = lat_lm.predict(2010)

         # Plot the 2010 prediction point

        plt.plot(lon_ten, lat_ten, '.', markersize=20, color=colors[i])

        # Plot the prediction line

        plt.plot(lon_pred, lat_pred, '--k')

        # Plot the cluster

        # plt.plot(sub_data['longitude'], sub_data['latitude'], '.', alpha=0.5, color=colors[i], label=i)

        

plt.title('Movement of Attacks by Sub Clusters of Clusters')
# Loop through each year.

for year in taliban['iyear'].unique():

    # Extract only they data from a given year.

    taliban_year = taliban[taliban['iyear'] == year]

    # Plot the year with it's appropriate alpha.

    # 1995 has a very small alpha do to it's insignificants.

    plt.plot(taliban_year['longitude'],taliban_year['latitude'], 'r.', alpha=1/(year%2000))

    

plt.title('Attack Alpha by Year')