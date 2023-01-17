import numpy as np

import pandas as pd



import geojson



import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap

import matplotlib.cm

from matplotlib.patches import Polygon

from matplotlib.collections import PatchCollection

import matplotlib.colors as colors



import seaborn as sns



cmap = sns.diverging_palette(220, 15, as_cmap=True)
# Reading Data

df_train = pd.read_csv('../input/favorita-grocery-sales-forecasting/train.csv')

df_items = pd.read_csv('../input/favorita-grocery-sales-forecasting/items.csv')

df_stores = pd.read_csv('../input/favorita-grocery-sales-forecasting/stores.csv')

df_train['date'] = pd.to_datetime(df_train['date'])

df_train.set_index('date', inplace=True)
# To make it more manageable, I'm just taking the data from the current year for the a further analysis

df_2017 = df_train[df_train.index>'2017-1-1']

df_2017 = pd.merge(df_2017, df_items, on='item_nbr', right_index=True)

df_2017 = pd.merge(df_2017, df_stores, on='store_nbr', right_index=True)
sns.set_style("whitegrid")



func = lambda df_grouped: len(df_grouped.unique())

df_monthly = df_train.groupby(pd.TimeGrouper('MS')).agg({'item_nbr': func,

                                                         'store_nbr': func})



ax = df_monthly.plot(subplots=True, layout=(1,2), figsize=(13,5), legend=False, linewidth=3, colormap=cmap)

ax[0][0].set_title("Favorita's Portfolio size Evolution")

ax[0][0].set_ylabel("Qtd. of different SKUs")

ax[0][1].set_title("Favorita's Growth")

ax[0][1].set_ylabel("Number of stores")

plt.show()
def func_tag(df):

    try:

        last_day = df.index.max()

        first_day = df.index.min()



        period_len = (last_day - first_day).days

        fill_rate = len(df) / period_len



        return pd.Series({'period_len': period_len, 'fill_rate': fill_rate})

    except:

        return pd.Series({'period_len': np.nan, 'fill_rate': np.nan})

    

df_train.drop(['id', 'unit_sales', 'onpromotion'], axis=1, inplace=True)

df_train = df_train.groupby(['store_nbr', 'item_nbr']).apply(func_tag).reset_index()

df_train['fill_rate'] = 100 * df_train['fill_rate']

df_train[df_train['fill_rate']>100] = 100



sns.set_style("whitegrid")

fig, axes = plt.subplots(nrows=2, figsize=(7, 12), sharex=False)



df_train[['period_len']].hist(ec='black', ax=axes[0])

axes[0].set_title('Distribution of the series length')

axes[0].set_xlabel('Days')



df_train[['fill_rate']].hist(ec='black', ax=axes[1])

axes[1].set_title('Distribution of the series fill rate')

axes[1].set_xlabel('Fill rate (%)')

plt.show()



del df_train
sns.set_style("white")

df_grouped_state = df_2017.groupby([pd.TimeGrouper('D'), 'state']).agg({'unit_sales':sum}).reset_index()



x = df_grouped_state.groupby('state').agg({'unit_sales': sum}).reset_index()

x['state'] = x['state'].apply(lambda state_name: state_name.upper())



with open("../input/ecuardorgeojson/ecuador.geojson") as json_file:

    json_data = geojson.load(json_file)



fig, ax = plt.subplots(figsize=(10,9))



patches = []

for feature in json_data['features']:

    state = feature['properties']['DPA_DESPRO'].upper()

    city = feature['properties']['DPA_DESCAN'].upper()

    poly = Polygon(np.array(feature['geometry']['coordinates'][0][0]), closed=True)

    patches.append({'state': state, 'city': city, 'poly': poly})

df_patches = pd.DataFrame.from_records(patches)

df_patches = pd.merge(df_patches, x, on='state', how='left')

df_patches = df_patches.fillna(1)



# cmap=matplotlib.cm.RdBu_r

p = PatchCollection(df_patches['poly'])#, cmap=matplotlib.cm.RdBu_r)



norm = colors.Normalize()

p.set_facecolor(cmap(norm(np.log(df_patches['unit_sales']))))



mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)

mapper.set_array(df_patches['unit_sales'])

plt.colorbar(mapper, shrink=0.4, label="2017's Sales Units (log)")



ax.add_collection(p)

ax.set_xlim(-81.2, -75)

ax.set_ylim(-5.1, 1.3)

plt.title('Heatmap - Total unit sales (Log Scale) per State', fontsize=14)

plt.tick_params(axis='both', which='both', bottom='off', top='off', labelleft='off', labelbottom='off')

plt.show()
sns.set_style("whitegrid")

fig, axes = plt.subplots(nrows=2, figsize=(7, 12), sharex=False)



df = df_grouped_state.pivot(index='date', columns='state', values='unit_sales').sum()

df.sort_values(ascending=True).plot(kind='barh', colormap=cmap, ax=axes[0])

axes[0].set_title('Total sales in 2017 - per State')

plt.xlabel('Total units sold')



df_grouped_city = df_2017.groupby('city').agg({'unit_sales':sum}).sort_values('unit_sales', ascending=True)

df_grouped_city.plot(kind='barh', colormap=cmap, ax=axes[1], legend=False)

axes[1].set_title('Total sales in 2017 - per City')

plt.xlabel('Total units sold')

plt.show()
ax = np.log(df_grouped_state.pivot(index='date', columns='state', values='unit_sales')).plot(figsize=(14,7), colormap=cmap, linewidth=2)

lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=12)

plt.title('Sales evolution - 2017', fontsize=14)

plt.ylabel('Total units sold')

plt.grid(which='minor')

plt.show()
df_2017['weekday'] = [item.weekday() for item in df_2017.index]

df_stores_weekday = df_2017.groupby(['store_nbr', 'weekday']).sum().reset_index()

df_stores_weekday = df_stores_weekday.pivot(index='weekday', columns='store_nbr', values='unit_sales')



plt.figure(figsize=(15,4))

ax = sns.heatmap(df_stores_weekday.apply(lambda col: (col-min(col))/(max(col)-min(col)), axis=0), 

                 cmap=cmap, cbar_kws={'label': 'Normalized Sale'})

ax.set_ylabel('Weekday')

ax.set_xlabel('Store number')

ax.set_yticklabels(['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN'])

ax.set_title('Sales intensity of each store - per weekday')

plt.yticks(rotation=0)

plt.show()
different_stores = df_stores_weekday.apply(lambda col: col.idxmax() if col.idxmax()<3 else None, axis=0).dropna().index

df_stores[df_stores['store_nbr'].isin(different_stores)].set_index('store_nbr')
df_train = pd.read_csv('../input/favorita-grocery-sales-forecasting/train.csv');

df_train['date'] = pd.to_datetime(df_train['date'])

df_train.set_index('date', inplace=True)



df = df_train[df_train.index>='2016-1-1'].drop(['id', 'store_nbr', 'onpromotion'], axis=1)

df = pd.merge(df, df_items[['item_nbr', 'class']], on='item_nbr', right_index=True).drop(['item_nbr'], axis=1)

df['month'] = [item.month for item in df.index]

df_item_month = df.groupby(['class', 'month']).mean().reset_index()

df_item_month = df_item_month.pivot(index='month', columns='class', values='unit_sales')



plt.figure(figsize=(10,15))

ax = sns.heatmap(df_item_month.apply(lambda col: (col-min(col))/(max(col)-min(col)), axis=0).T, 

                 cmap=cmap, cbar_kws={'label': 'Normalized Sale'})

ax.set_ylabel('Product Class')

ax.set_xlabel('Month')

ax.set_title("Sales intensity of each product's class - per month")

plt.show()
sns.set_style("whitegrid")

df = df_2017.groupby('family').agg({'unit_sales': sum}).sort_values('unit_sales', ascending=False)



OTHERS = df.iloc[10:].sum()

df.drop(df.iloc[10:].index.tolist(), inplace=True)

df.loc['OTHERS'] = OTHERS



ax = df.plot.pie(y='unit_sales', figsize=(6, 6), colormap=cmap)

handles, labels = ax.get_legend_handles_labels()

lgd = ax.legend(handles, labels, bbox_to_anchor=(1.3, 0.8), loc=2, borderaxespad=0., fontsize=12)

plt.ylabel(' ')

plt.title('Market share per product family (2017)', fontsize=14)

plt.show()
sns.set_style("whitegrid")

df = df_2017.groupby('class').agg({'unit_sales': sum}).sort_values('unit_sales', ascending=False)



OTHERS = df.iloc[10:].sum()

df.drop(df.iloc[10:].index.tolist(), inplace=True)

df.loc['OTHERS'] = OTHERS



ax = df.plot.pie(y='unit_sales', figsize=(6, 6), colormap=cmap)

handles, labels = ax.get_legend_handles_labels()

lgd = ax.legend(handles, labels, bbox_to_anchor=(1.3, 0.8), loc=2, borderaxespad=0., fontsize=12)

plt.ylabel(' ')

plt.title('Market share per product class (2017)', fontsize=14)

plt.show()
func_norm = lambda df: (df['unit_sales'] - df['unit_sales'].min())/(df['unit_sales'].max() - df['unit_sales'].min())

df_2017 = df_2017[['unit_sales', 'item_nbr', 'store_nbr', 'onpromotion']].set_index(['onpromotion'])

df_2017 = df_2017[df_2017['unit_sales']>0] # Take just positive sales

df_2017 =  df_2017.groupby(['store_nbr', 'item_nbr']).apply(func_norm).reset_index() # Normalize sales item-store



def func(df):

    if len(df['onpromotion'].unique())==2:

        return df

    else:

        pass

    

teste = df_2017.groupby(['item_nbr', 'store_nbr']).apply(func).dropna()



sns.boxplot(x="onpromotion", y="unit_sales", data=teste)

plt.ylabel('unit_sales normalized')

plt.show()