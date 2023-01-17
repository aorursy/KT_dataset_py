import pandas as pd
beer_data=pd.read_csv('../input/beers.csv',sep=',')
# Replace NaN values with 0 in ibu (bitterness)

beer_data=beer_data.fillna(value=0)
brewery_data=pd.read_csv('../input/breweries.csv',sep=',')
states_breweries=brewery_data.groupby(['state']).count()

states_breweries_top=states_breweries.sort(columns='name', axis=0, ascending=False)

states_breweries_top.head()
cities_breweries=brewery_data.groupby(['city']).count()

cities_breweries_top=cities_breweries.sort(columns='name', axis=0, ascending=False)

cities_breweries_top.head()
beer_style=beer_data.groupby(['style']).count()

beer_style_top=beer_style.sort(columns='id', axis=0, ascending=False)

beer_style_top.head()
beer_mean=beer_data['abv'].mean(axis=0)

beer_mean
beer_median=beer_data['abv'].median(axis=0)

beer_median
beer_abv_top=beer_data.sort(columns='abv', axis=0, ascending=False)

beer_abv_top.head()