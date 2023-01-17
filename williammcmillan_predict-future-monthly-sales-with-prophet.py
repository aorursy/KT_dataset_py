# Basic packages

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import random as rd # generating random numbers

import datetime # manipulating date formats

# Viz

import matplotlib.pyplot as plt # basic plotting

import seaborn as sns # for prettier plots



from itertools import product

from sklearn.preprocessing import LabelEncoder



import gc; gc.enable()



# TIME SERIES

from statsmodels.tsa.arima_model import ARIMA

from statsmodels.tsa.statespace.sarimax import SARIMAX

from pandas.plotting import autocorrelation_plot

from statsmodels.tsa.stattools import adfuller, acf, pacf,arma_order_select_ic

import statsmodels.formula.api as smf

import statsmodels.tsa.api as smt

import statsmodels.api as sm

from scipy import stats





# settings

import warnings

warnings.filterwarnings("ignore")



pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 100)
folder_path = '../input/competitive-data-science-predict-future-sales/'
# Load item_categories, items, shops & train (test & submission data will not be needed).

item_cat = pd.read_csv(f'{folder_path}item_categories.csv')

items = pd.read_csv(f'{folder_path}items.csv')

shops = pd.read_csv(f'{folder_path}shops.csv')

#submission = pd.read_csv(f'{folder_path}sample_submission.csv')

#test = pd.read_csv(f'{folder_path}test.csv')

train = pd.read_csv(f'{folder_path}sales_train.csv')
item_cat.head(10)
items.head(10)
shops.head(10)
train.head(10)
# Join the tables to the train data set.



train = pd.merge(train, items, on='item_id', how='inner')

train = pd.merge(train, item_cat, on='item_category_id', how='inner')

train = pd.merge(train, shops, on='shop_id', how='inner')



del item_cat, shops



gc.collect()
print(train.shape)

train.head()
train['item_price'].sort_values().tail()
train['item_cnt_day'].sort_values().tail()
train.loc[train['item_price'] > 300000.00]
# Plotting the two outliers highlights their respective ridiculousness.

plt.figure(figsize=(10,4))

plt.xlim(-100, 2500)

sns.boxplot(x=train.item_cnt_day)



plt.figure(figsize=(10,4))

plt.xlim(train.item_price.min(), train.item_price.max()*1.1)

sns.boxplot(x=train.item_price)
# Dropping two outlier values highlighted above:

train = train[train.item_price<300000]

train = train[train.item_cnt_day<2000]
train.shape
# Create the correct median_price value based on other positively valued purchases with the same 'date_block_num', 'shop_id', & 'item_id':

median_price = train[(train['date_block_num']==4)&(train['shop_id']==32)&(train['item_id']==2973)&(train['item_price']>0)].item_price.median()



# Change the 'item_price' value for the entry with a negative value

train.loc[train['item_price']<0, 'item_price'] = median_price
# Make sure the value has been changed

train[(train['shop_id']==32)&(train['item_id']==2973)&(train['date_block_num']==4)].head()
# A few shops are duplicates of each other (according to its name). Both the train and test sets need to be fixed.

# Якутск Орджоникидзе, 56

train.loc[train.shop_id == 0, 'shop_id'] = 57

#test.loc[test.shop_id == 0, 'shop_id'] = 57

# Якутск ТЦ "Центральный"

train.loc[train.shop_id == 1, 'shop_id'] = 58

#test.loc[test.shop_id == 1, 'shop_id'] = 58

# Жуковский ул. Чкалова 39м²

train.loc[train.shop_id == 10, 'shop_id'] = 11

#test.loc[test.shop_id == 10, 'shop_id'] = 11



# An extra space on a particular shope name needs to be removed in order to set up execution of the split() method.

train.loc[train.shop_name == 'Сергиев Посад ТЦ "7Я"', 'shop_name'] = 'СергиевПосад ТЦ "7Я"'
dict_categories = ['Cinema - DVD', 'PC Games - Standard Editions',

                    'Music - Local Production CD', 'Games - PS3', 'Cinema - Blu-Ray',

                    'Games - XBOX 360', 'PC Games - Additional Editions', 'Games - PS4',

                    'Gifts - Stuffed Toys', 'Gifts - Board Games (Compact)',

                    'Gifts - Figures', 'Cinema - Blu-Ray 3D',

                    'Programs - Home and Office', 'Gifts - Development',

                    'Gifts - Board Games', 'Gifts - Souvenirs (on the hinge)',

                    'Cinema - Collection', 'Music - MP3', 'Games - PSP',

                    'Gifts - Bags, Albums, Mouse Pads', 'Gifts - Souvenirs',

                    'Books - Audiobooks', 'Gifts - Gadgets, robots, sports',

                    'Accessories - PS4', 'Games - PSVita',

                    'Books - Methodical materials 1C', 'Payment cards - PSN',

                    'PC Games - Digit', 'Games - Game Accessories', 'Accessories - XBOX 360',

                    'Accessories - PS3', 'Games - XBOX ONE', 'Music - Vinyl',

                    'Programs - 1C: Enterprise 8', 'PC Games - Collectible Editions',

                    'Gifts - Attributes', 'Service Tools',

                    'Music - branded production CD', 'Payment cards - Live!',

                    'Game consoles - PS4', 'Accessories - PSVita', 'Batteries',

                    'Music - Music Video', 'Game Consoles - PS3',

                    'Books - Comics, Manga', 'Game Consoles - XBOX 360',

                    'Books - Audiobooks 1C', 'Books - Digit',

                    'Payment cards (Cinema, Music, Games)', 'Gifts - Cards, stickers',

                    'Accessories - XBOX ONE', 'Pure media (piece)',

                    'Programs - Home and Office (Digital)', 'Programs - Educational',

                    'Game consoles - PSVita', 'Books - Artbooks, encyclopedias',

                    'Programs - Educational (Digit)', 'Accessories - PSP',

                    'Gaming consoles - XBOX ONE', 'Delivery of goods',

                    'Payment Cards - Live! (Figure) ',' Tickets (Figure) ',

                    'Music - Gift Edition', 'Service Tools - Tickets',

                    'Net media (spire)', 'Cinema - Blu-Ray 4K', 'Game consoles - PSP',

                    'Game Consoles - Others', 'Books - Audiobooks (Figure)',

                    'Gifts - Certificates, Services', 'Android Games - Digit',

                    'Programs - MAC (Digit)', 'Payment Cards - Windows (Digit)',

                    'Books - Business Literature', 'Games - PS2', 'MAC Games - Digit',

                    'Books - Computer Literature', 'Books - Travel Guides',

                    'PC - Headsets / Headphones', 'Books - Fiction',

                    'Books - Cards', 'Accessories - PS2', 'Game consoles - PS2',

                    'Books - Cognitive literature']



dict_shops = ['Moscow Shopping Center "Semenovskiy"', 

              'Moscow TRK "Atrium"', 

              "Khimki Shopping Center",

              'Moscow TC "MEGA Teply Stan" II', 

              'Yakutsk Ordzhonikidze, 56',

              'St. Petersburg TC "Nevsky Center"', 

              'Moscow TC "MEGA Belaya Dacha II"',

              'Voronezh (Plekhanovskaya, 13)', 

              'Yakutsk Shopping Center "Central"',

              'Chekhov SEC "Carnival"', 

              'Sergiev Posad TC "7Ya"',

              'Tyumen TC "Goodwin"',

              'Kursk TC "Pushkinsky"', 

              'Kaluga SEC "XXI Century"',

              'N.Novgorod Science and entertainment complex "Fantastic"',

              'Moscow MTRC "Afi Mall"',

              'Voronezh SEC "Maksimir"', 'Surgut SEC "City Mall"',

              'Moscow Shopping Center "Areal" (Belyaevo)', 'Krasnoyarsk Shopping Center "June"',

              'Moscow TK "Budenovsky" (pav.K7)', 'Ufa "Family" 2',

              'Kolomna Shopping Center "Rio"', 'Moscow Shopping Center "Perlovsky"',

              'Moscow Shopping Center "New Century" (Novokosino)', 'Omsk Shopping Center "Mega"',

              'Moscow Shop C21', 'Tyumen Shopping Center "Green Coast"',

              'Ufa TC "Central"', 'Yaroslavl shopping center "Altair"',

              'RostovNaDonu "Mega" Shopping Center', '"Novosibirsk Mega "Shopping Center',

              'Samara Shopping Center "Melody"', 'St. Petersburg TC "Sennaya"',

              "Volzhsky Shopping Center 'Volga Mall' ",

              'Vologda Mall "Marmelad"', 'Kazan TC "ParkHouse" II',

              'Samara Shopping Center ParkHouse', '1C-Online Digital Warehouse',

              'Online store of emergencies', 'Adygea Shopping Center "Mega"',

              'Balashikha shopping center "October-Kinomir"' , 'Krasnoyarsk Shopping center "Vzletka Plaza" ',

              'Tomsk SEC "Emerald City"', 'Zhukovsky st. Chkalov 39m? ',

              'Kazan Shopping Center "Behetle"', 'Tyumen SEC "Crystal"',

              'RostovNaDonu TRK "Megacenter Horizon"',

              '! Yakutsk Ordzhonikidze, 56 fran', 'Moscow TC "Silver House"',

              'Moscow TK "Budenovsky" (pav.A2)', "N.Novgorod SEC 'RIO' ",

              '! Yakutsk TTS "Central" fran', 'Mytishchi TRK "XL-3"',

              'RostovNaDonu TRK "Megatsentr Horizon" Ostrovnoy', 'Exit Trade',

              'Voronezh SEC City-Park "Grad"', "Moscow 'Sale'",

              'Zhukovsky st. Chkalov 39m² ',' Novosibirsk Shopping Mall "Gallery Novosibirsk"']
train.item_category_name = train.item_category_name.map(dict(zip(train.item_category_name.value_counts().index, dict_categories)))

train.shop_name = train.shop_name.map(dict(zip(train.shop_name.value_counts().index, dict_shops)))
train['shop_name'] = train['shop_name'].replace([' Novosibirsk Shopping Mall "Gallery Novosibirsk"'], 'Novosibirsk Shopping Mall "Gallery Novosibirsk"')

train['shop_name'] = train['shop_name'].replace(['! Yakutsk Ordzhonikidze, 56 fran'], 'Yakutsk Ordzhonikidze, 56 fran')

train['shop_name'] = train['shop_name'].replace(['! Yakutsk TTS "Central" fran   '], 'Yakutsk TTS "Central" fran')

train['shop_name'] = train['shop_name'].replace(['! Yakutsk TTS "Central" fran'], 'Yakutsk TTS "Central" fran')

train['shop_name'] = train['shop_name'].replace(['1C-Online Digital Warehouse'], 'Online Digital Warehouse')

train['shop_name'] = train['shop_name'].replace(['"Novosibirsk Mega "Shopping Center'], 'Novosibirsk "Mega" Shopping Center')
sales = train.copy()



sales['city'] = sales['shop_name'].str.split(' ').map(lambda x: x[0])

sales.loc[sales.city == '!Якутск', 'city'] = 'Якутск'

##sales['city_code'] = LabelEncoder().fit_transform(sales['city'])



sales['split'] = sales['item_category_name'].str.split('-')

sales['type'] = sales['split'].map(lambda x: x[0].strip())

##sales['type_code'] = LabelEncoder().fit_transform(sales['type'])



## if subtype is nan then type

sales['subtype'] = sales['split'].map(lambda x: x[1].strip() if len(x) > 1 else x[0].strip())

##sales['subtype_code'] = LabelEncoder().fit_transform(sales['subtype'])

sales = sales.drop(columns=['shop_name', 'split'], inplace=False)

sales.drop(['item_name'], axis=1, inplace=True)



sales.drop(columns=['item_category_name'], axis=1, inplace=True)
train['city'] = train['shop_name'].str.split(' ').map(lambda x: x[0])

train.loc[train.city == '!Якутск', 'city'] = 'Якутск'

train['city_code'] = LabelEncoder().fit_transform(train['city'])



train['split'] = train['item_category_name'].str.split('-')

train['type'] = train['split'].map(lambda x: x[0].strip())

train['type_code'] = LabelEncoder().fit_transform(train['type'])



# if subtype is nan then type

train['subtype'] = train['split'].map(lambda x: x[1].strip() if len(x) > 1 else x[0].strip())

train['subtype_code'] = LabelEncoder().fit_transform(train['subtype'])

train = train.drop(columns=['item_category_name', 'shop_name', 'split', 'city', 'type', 'subtype'], inplace=False)



train.drop(['item_name'], axis=1, inplace=True)
sales.rename(columns={'date_block_num': 'month'}, inplace=True)

train.rename(columns={'date_block_num': 'month'}, inplace=True)
sales['type'] = sales['type'].replace(['PC Games', 'Android Games', 'MAC Games','PC', 'Accessories', 'Batteries'], 'Games')

sales['type'] = sales['type'].replace(['Game consoles', 'Gaming consoles'], 'Game Consoles')

sales['type'] = sales['type'].replace(['Payment cards', 'Payment cards (Cinema, Music, Games)'], 'Payment Cards')

sales['type'] = sales['type'].replace(['Games', 'Game Consoles'], 'Gaming')

sales['type'] = sales['type'].replace(['Cinema'], 'Movies')

sales['type'] = sales['type'].replace(['Net media (spire)', 'Pure media (piece)', 'Tickets (Figure)', 'Programs'], 'Software')

sales['type'] = sales['type'].replace(['Service Tools', 'Delivery of goods'], 'Service & Delivery')

sales['type'] = sales['type'].replace(['Payment Cards', 'Gifts'], 'Gifts & Gift Cards')

sales['type'].value_counts()
sales['total_sales'] = (sales['item_price'] * sales['item_cnt_day'])/1000

train['total_sales'] = (train['item_price'] * train['item_cnt_day'])/1000
year_sales = sales.copy()
year_sales['year'] = sales['date'].str.split('.').map(lambda x: x[2])

year_sales.drop(columns=['date'], axis=1, inplace=True)
#formatting the date column correctly

sales.date = sales.date.apply(lambda x:datetime.datetime.strptime(x, '%d.%m.%Y'))

# check

print(sales.info())
# type_sales = sales.groupby(["shop_id","type"])[

#     "year","item_price","item_cnt_day"].agg({"year":["min",'max'],"item_price":"mean","item_cnt_day":"sum"})
#formatting the date column correctly

train.date = train.date.apply(lambda x:datetime.datetime.strptime(x, '%d.%m.%Y'))

# check

print(train.info())
# top_ten = top_ten_df.groupby(["date_block_num","shop_id","item_id"])[

#     "date","item_price","item_cnt_day"].agg({"date":["min",'max'],"item_price":"mean","item_cnt_day":"sum"})
# Aggregate to monthly level the required metrics



monthly_sales = train.groupby(["month","shop_id","item_id"])[

    "date","item_price","item_cnt_day"].agg({"date":["min",'max'],"item_price":"mean","item_cnt_day":"sum"})



## Lets break down the line of code here:

# Aggregate by "date-block_num" (month), "shop_id", and "item_id"

# Select the columns: "date", "item_price", and "item_cnt" (sales)

# Provide a dictionary which says what aggregation to perform on which column

# min and max on the date

# average of the item_price

# sum of the sales



monthly_sales.head()
# Number of items per category



plt.style.use('fivethirtyeight')

x = items.groupby(['item_category_id']).count()

x = x.sort_values(by='item_id',ascending=False)

x = x.iloc[0:10].reset_index()

x

# plot

plt.figure(figsize=(8,4))

ax = sns.barplot(x.item_category_id, x.item_id)

plt.title("Items per Category")

plt.ylabel('# of Items', fontsize=12)

plt.xlabel('Category', fontsize=12)

plt.show()
year_sales['month'] = year_sales['month'].replace([11, 23], 'Dec')

year_sales['month'] = year_sales['month'].replace([10, 22], 'Nov')

year_sales['month'] = year_sales['month'].replace([9, 21, 33], 'Oct')

year_sales['month'] = year_sales['month'].replace([8, 20, 32], 'Sept')

year_sales['month'] = year_sales['month'].replace([7, 19, 31], 'Aug')

year_sales['month'] = year_sales['month'].replace([6, 18, 30], 'July')

year_sales['month'] = year_sales['month'].replace([5, 17, 29], 'June')

year_sales['month'] = year_sales['month'].replace([4, 16, 28], 'May')

year_sales['month'] = year_sales['month'].replace([3, 15, 27], 'Apr')

year_sales['month'] = year_sales['month'].replace([2, 14, 26], 'Mar')

year_sales['month'] = year_sales['month'].replace([1, 13, 25], 'Feb')

year_sales['month'] = year_sales['month'].replace([0, 12, 24], 'Jan')
sales['month'] = sales['month'] + 1

# train['month'] = train['month'] + 1
sales.rename(columns={'type': 'category'}, inplace=True)

year_sales.rename(columns={'type': 'category'}, inplace=True)

train.rename(columns={'type': 'category'}, inplace=True)
# Computing Total Sales of the Company per month & plotting that data:



plt.style.use('fivethirtyeight')



cs = train.groupby(["month"])["total_sales"].sum()

cs.astype('float')

plt.figure(figsize=(14,7))

plt.title('1C Company Sales (January 2013 - October 2015)', fontsize=22)

plt.xlabel('Month & Year', fontsize=14)

plt.ylabel('Sales/1000 in Russian Rubles', fontsize=14)

ticks = [0, 4, 10, 16, 22, 28, 33]



plt.xticks(ticks, ['January13', 'May13', 'November13', 'May14', 'November14', 'May15', 'October15'])

           

plt.plot(cs)

plt.show();
# Assessing the presence of trend & seasonality via Rolling Statistics:



plt.style.use("fivethirtyeight")

plt.figure(figsize=(14,7))

plt.plot(cs.rolling(window=12,center=False).mean(),label='Rolling Mean - [Trend Line of Company Sales]');

plt.plot(cs.rolling(window=12,center=False).std(),label='Rolling Standard Deviation - [Seasonality]');

plt.legend()

plt.show();
# Decomposing Time-Series into Trend, Seasonality, and Residuals

# Is the seasonality Multiplicative or Additive? 

# That is, does the magnitude of the seasonality increase when the time series increases?

# Let's check for Multiplicative Seasonality first with statsmodels.tsa (Time Series Analysis):



import statsmodels.api as sm

plt.style.use('seaborn-poster')



ts_decomposed = sm.tsa.seasonal_decompose(cs.values,freq=12,model="multiplicative")

plt.figure(figsize=(16,12))

fig = ts_decomposed.plot()

fig.show()
year_sales.head()
year_sales['month'] = year_sales['month'].replace(['Dec'], 12)

year_sales['month'] = year_sales['month'].replace(['Nov'], 11)

year_sales['month'] = year_sales['month'].replace(['Oct'], 10)

year_sales['month'] = year_sales['month'].replace(['Sept'], 9)

year_sales['month'] = year_sales['month'].replace(['Aug'], 8)

year_sales['month'] = year_sales['month'].replace(['July'], 7)

year_sales['month'] = year_sales['month'].replace(['June'], 6)

year_sales['month'] = year_sales['month'].replace(['May'], 5)

year_sales['month'] = year_sales['month'].replace(['Apr'], 4)

year_sales['month'] = year_sales['month'].replace(['Mar'], 3)

year_sales['month'] = year_sales['month'].replace(['Feb'], 2)

year_sales['month'] = year_sales['month'].replace(['Jan'], 1)
year_sales.rename(columns={'month': 'Month'}, inplace=True)

year_sales.rename(columns={'total_sales': 'Sales'}, inplace=True)
year_sales.head()
plt.style.available
import seaborn as sns

from seaborn import FacetGrid



plt.style.use('seaborn-darkgrid')



monthlyRev = pd.DataFrame(year_sales.groupby(["Month", "year"], as_index=False)["Sales"].sum())



g = sns.FacetGrid(data = monthlyRev.sort_values(by="Month"), hue = "year", size = 7, legend_out=True)

g = g.map(plt.plot, "Month", "Sales")

g.add_legend()

plt.subplots_adjust(top=0.9)

g.fig.suptitle('1c - Annual Company Sales', fontsize=22)

g.set(xticks=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

g.set_xticklabels(['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'], rotation=35)

g

plt.show();
year_sales.head()
category_sales = sales[['month', 'category', 'total_sales']].copy()

bar_cat_sales = sales[['month', 'category', 'total_sales']].copy()

z = bar_cat_sales.groupby(["category"])["total_sales"].agg({'total_sales': sum})

z = z.sort_values(by='total_sales',ascending=False)

z = z.iloc[0:10].reset_index()

z
import seaborn as sns

plt.style.use('fivethirtyeight')



sns.barplot(y = z.category, x = "total_sales", data = z)

plt.title("Sales Broken Down by Category")

plt.xlabel("Sales")

plt.ylabel("Product Category")

plt.show();
item_cat_sales = sales[['month', 'item_category_id', 'category', 'subtype','item_cnt_day']].copy()



item_category_sales = item_cat_sales.groupby(["item_category_id", 'category', 'subtype'])["item_cnt_day"].agg({'item_cnt_day': sum})

# item_category_sales['cat_sub'] = item_category_sales['category'] + item_category_sales['subtype']

z = item_category_sales.sort_values(by='item_cnt_day', ascending=False)

z = z.iloc[0:10].reset_index()

z
import seaborn as sns

plt.style.use('fivethirtyeight')



sns.barplot(y = z.subtype, x = "item_cnt_day", data = z)

plt.title("Highest Number of Units")

plt.xlabel("Number of Units")

plt.ylabel("Category Subtype")

plt.show();
annual_sales = year_sales[['Month', 'year', 'category', 'Sales']].copy()

annual_sales.head()
annual_movies = annual_sales[annual_sales.category=='Movies']

annual_music = annual_sales[annual_sales.category=='Music']

annual_software = annual_sales[annual_sales.category=='Software']



annual_eliminate = annual_sales.loc[(annual_sales.category == 'Movies')|(annual_sales.category == 'Music')|(annual_sales.category == 'Programs')]



annual_gaming = annual_sales[annual_sales.category=='Gaming']



annual_gifts = annual_sales[annual_sales.category=='Gifts & Gift Cards']

annual_books = annual_sales[annual_sales.category=='Books']

#annual_cards = annual_sales[annual_sales.category=='Payment Cards']

annual_service = annual_sales[annual_sales.category=='Service & Delivery']



annual_keep = annual_sales.loc[(annual_sales.category=='Gifts & Gift Cards')|(annual_sales.category=='Books')|(annual_sales.category=='Service & Delivery')]
# Movies decreasing each year

monthlySales = pd.DataFrame(annual_movies.groupby(["Month", "year"], as_index=False)["Sales"].sum())



plt.style.use('fivethirtyeight')

g = sns.FacetGrid(data = monthlySales.sort_values(by="Month"), hue = "year", size = 7, legend_out=True)

g = g.map(plt.plot, "Month", "Sales")

g.add_legend()

plt.subplots_adjust(top=0.9)

g.fig.suptitle('Annual Movie Sales', fontsize=22)

g;
# Music decreasing each year

monthlySales = pd.DataFrame(annual_music.groupby(["Month", "year"], as_index=False)["Sales"].sum())



g = sns.FacetGrid(data = monthlySales.sort_values(by="Month"), hue = "year", size = 7, legend_out=True)

g = g.map(plt.plot, "Month", "Sales")

g.add_legend()

plt.subplots_adjust(top=0.9)

g.fig.suptitle('Annual Music Sales', fontsize=22)

g;
# Programs decreasing each year

monthlySales = pd.DataFrame(annual_software.groupby(["Month", "year"], as_index=False)["Sales"].sum())



g = sns.FacetGrid(data = monthlySales.sort_values(by="Month"), hue = "year", size = 7, legend_out=True)

g = g.map(plt.plot, "Month", "Sales")

g.add_legend()

plt.subplots_adjust(top=0.9)

g.fig.suptitle('Annual Software Sales', fontsize=22)

g;
# Eliminate categories - annual

monthlySales = pd.DataFrame(annual_eliminate.groupby(["Month", "year"], as_index=False)["Sales"].sum())



g = sns.FacetGrid(data = monthlySales.sort_values(by="Month"), hue = "year", size = 7, legend_out=True)

g = g.map(plt.plot, "Month", "Sales")

g.add_legend()

plt.subplots_adjust(top=0.9)

g.fig.suptitle('Annual Sales - Movies, Music, Software', fontsize=22)

g;
# Gaming each year

monthlySales = pd.DataFrame(annual_gaming.groupby(["Month", "year"], as_index=False)["Sales"].sum())



g = sns.FacetGrid(data = monthlySales.sort_values(by="Month"), hue = "year", size = 7, legend_out=True)

g = g.map(plt.plot, "Month", "Sales")

g.add_legend()

plt.subplots_adjust(top=0.9)

g.fig.suptitle('Annual Gaming Sales', fontsize=22)

g;
# Gifts increasing each year

monthlySales = pd.DataFrame(annual_gifts.groupby(["Month", "year"], as_index=False)["Sales"].sum())



g = sns.FacetGrid(data = monthlySales.sort_values(by="Month"), hue = "year", size = 7, legend_out=True)

g = g.map(plt.plot, "Month", "Sales")

g.add_legend()

plt.subplots_adjust(top=0.9)

g.fig.suptitle('Annual Gifts Sales', fontsize=22)

g;
# Books increasing each year

monthlySales = pd.DataFrame(annual_books.groupby(["Month", "year"], as_index=False)["Sales"].sum())



g = sns.FacetGrid(data = monthlySales.sort_values(by="Month"), hue = "year", size = 7, legend_out=True)

g = g.map(plt.plot, "Month", "Sales")

g.add_legend()

plt.subplots_adjust(top=0.9)

g.fig.suptitle('Annual Book Sales', fontsize=22)

g;
# Services & delivery increasing each year

monthlySales = pd.DataFrame(annual_service.groupby(["Month", "year"], as_index=False)["Sales"].sum())



g = sns.FacetGrid(data = monthlySales.sort_values(by="Month"), hue = "year", size = 7, legend_out=True)

g = g.map(plt.plot, "Month", "Sales")

g.add_legend()

plt.subplots_adjust(top=0.9)

g.fig.suptitle('Services & Delivery - Annual', fontsize=22)

g;
# Keep - Annual

monthlySales = pd.DataFrame(annual_keep.groupby(["Month", "year"], as_index=False)["Sales"].sum())



g = sns.FacetGrid(data = monthlySales.sort_values(by="Month"), hue = "year", size = 7, legend_out=True)

g = g.map(plt.plot, "Month", "Sales")

g.add_legend()

plt.subplots_adjust(top=0.9)

g.fig.suptitle('Annual Sales: Books, Gifts & Gift Cards, Service & Delivery', fontsize=22)

g;
category_sales['month'] = category_sales['month'] - 1
eliminate_1 = category_sales[category_sales.category=='Movies']

eliminate_2 = category_sales[category_sales.category=='Music']

eliminate_3 = category_sales[category_sales.category=='Software']



eliminate = category_sales.loc[(category_sales.category == 'Movies')|(category_sales.category == 'Music')|(category_sales.category == 'Software')]



keep_1 = category_sales[category_sales.category=='Gaming']



keep_2 = category_sales[category_sales.category=='Gifts & Gift Cards']

keep_3 = category_sales[category_sales.category=='Books']

keep_4 = category_sales[category_sales.category=='Service & Delivery']



keep = category_sales.loc[(category_sales.category=='Gifts & Gift Cards')|(category_sales.category=='Books')|(category_sales.category=='Service & Delivery')]
# Gaming plotted against Total Company Sales



ts = keep_1.groupby(["month"])["total_sales"].sum()

ts.astype('float')

ax = cs.plot(figsize=(14,7), title="Gaming Sales in Relation to Total Company Sales (January 2013 - October 2015)", fontsize=16, legend=True)

ts.plot(ax=ax)



plt.xlabel('Time - (month #)', fontsize=14)

plt.ylabel('Sales', fontsize=14)

ax.legend(["Total Company Sales", "Gaming"])

plt.show();
# All 8 Categories Plotted Together



ts1 = keep_1.groupby(["month"])["total_sales"].sum()

ts2 = keep_2.groupby(["month"])["total_sales"].sum()

ts3 = keep_3.groupby(["month"])["total_sales"].sum()

ts4 = keep_4.groupby(["month"])["total_sales"].sum()

ts5 = eliminate_1.groupby(["month"])["total_sales"].sum()

ts6 = eliminate_2.groupby(["month"])["total_sales"].sum()

ts7 = eliminate_3.groupby(["month"])["total_sales"].sum()



ts.astype('float')

ax = cs.plot(figsize=(14,7), title="Total Sales & Sales by Category (January 2013 - October 2015)", fontsize=14, legend=True)

ts1.plot(ax=ax)

ts2.plot(ax=ax)

ts3.plot(ax=ax)

ts4.plot(ax=ax)

ts5.plot(ax=ax)

ts6.plot(ax=ax)

ts7.plot(ax=ax)

#plt.figure(figsize=(14,7))

#plt.title('Total Sales Gaming')

plt.xlabel('Month', fontsize=14)

plt.ylabel('Sales', fontsize=14)

ax.legend(["Total Sales","Gaming", "Gifts & Gift Cards", "Books", "Service & Delivery", "Movies", "Music", "Software"]);

ticks = [0, 4, 10, 16, 22, 28, 33]



plt.xticks(ticks, ['January13', 'May13', 'November13', 'May14', 'November14', 'May15', 'October15'])



plt.show();
# Gaming Removed from Category Plots



#ts1 = keep_1.groupby(["month"])["total_sales"].sum()

ts2 = keep_2.groupby(["month"])["total_sales"].sum()

ts3 = keep_3.groupby(["month"])["total_sales"].sum()

ts4 = keep_4.groupby(["month"])["total_sales"].sum()

ts5 = eliminate_1.groupby(["month"])["total_sales"].sum()

ts6 = eliminate_2.groupby(["month"])["total_sales"].sum()

ts7 = eliminate_3.groupby(["month"])["total_sales"].sum()



ts.astype('float')

ax = ts2.plot(figsize=(14,7), title="Sales of 1C Company (Jan13 - Oct15) ~ Gaming Removed", fontsize=14, legend=True)

#ts2.plot(ax=ax)

ts3.plot(ax=ax)

ts4.plot(ax=ax)

ts5.plot(ax=ax)

ts6.plot(ax=ax)

ts7.plot(ax=ax)

plt.xlabel('Month', fontsize=14)

plt.ylabel('Sales', fontsize=14)

ax.legend(["Gifts & Gift Cards", "Books", "Service & Delivery", "Movies", "Music", "Software"])

ticks = [0, 4, 10, 16, 22, 28, 33]



plt.xticks(ticks, ['January13', 'May13', 'November13', 'May14', 'November14', 'May15', 'October15'])

           

plt.show();
ts = keep_2.groupby(["month"])["total_sales"].sum()

ts.astype('float')

plt.figure(figsize=(14,7))

plt.title('Total Sales Gifts & Gift Cards')

plt.xlabel('Month #')

plt.ylabel('Sales')

plt.plot(ts);
ts = keep_3.groupby(["month"])["total_sales"].sum()

ts.astype('float')

plt.figure(figsize=(14,7))

plt.title('Total Sales Books')

plt.xlabel('Month #)')

plt.ylabel('Sales')

plt.plot(ts);
ts = keep_4.groupby(["month"])["total_sales"].sum()

ts.astype('float')

plt.figure(figsize=(14,7))

plt.title('Service & Delivery Total Sales')

plt.xlabel('Month #)')

plt.ylabel('Sales')

plt.plot(ts);
ts = eliminate_1.groupby(["month"])["total_sales"].sum()

ts.astype('float')

plt.figure(figsize=(14,7))

plt.title('Total Sales Movies')

plt.xlabel('Month #')

plt.ylabel('Sales')

plt.plot(ts);
ts = eliminate_2.groupby(["month"])["total_sales"].sum()

ts.astype('float')

plt.figure(figsize=(14,7))

plt.title('Total Sales Music')

plt.xlabel('Month #')

plt.ylabel('Sales')

plt.plot(ts);
ts = eliminate_3.groupby(["month"])["total_sales"].sum()

ts.astype('float')

plt.figure(figsize=(14,7))

plt.title('Software')

plt.xlabel('Month #')

plt.ylabel('Sales')

plt.plot(ts);
# Plotting Those in 'Keep' with Gaming Removed [Gifts, Books, Service & Delivery]



#ts1 = keep_1.groupby(["month"])["total_sales"].sum()

ts2 = keep_2.groupby(["month"])["total_sales"].sum()

ts3 = keep_3.groupby(["month"])["total_sales"].sum()

ts4 = keep_4.groupby(["month"])["total_sales"].sum()



ts.astype('float')

ax = ts2.plot(figsize=(14,7), title="Gift, Books & Service Sales (Jan13 - Oct15)", fontsize=14, legend=True)

#ts2.plot(ax=ax)

ts3.plot(ax=ax)

ts4.plot(ax=ax)

plt.xlabel('Month #', fontsize=14)

plt.ylabel('Sales', fontsize=14)

ax.legend(["Gifts & Gift Cards", "Books", "Service & Delivery"]);

plt.show();
# Plotting 'Keep' versus 'Eliminate'



ts1 = keep.groupby(["month"])["total_sales"].sum()

ts2 = eliminate.groupby(["month"])["total_sales"].sum()



ts.astype('float')

ax = ts1.plot(figsize=(14,7), title="'Retain' Compared to 'Relinquish' - 1C Company (January 2013 - October 2015)", fontsize=14, legend=True)

ts2.plot(ax=ax)



plt.xlabel('Month ', fontsize=14)

plt.ylabel('Sales', fontsize=14)

ax.legend(["Retain: Gifts, Books, Service","Relinquish: Movies, Music, Software"]);

ticks = [0, 4, 10, 16, 22, 28, 33]



plt.xticks(ticks, ['January13', 'May13', 'November13', 'May14', 'November14', 'May15', 'October15'])



plt.show();
keepers = keep.groupby(["month"])["total_sales"].sum()

keepers.astype('float')

plt.figure(figsize=(14,7))

plt.title('Total Sales Keepers')

plt.xlabel('Time - (month #)')

plt.ylabel('Sales (# of units)')

plt.plot(keepers);
shed = eliminate.groupby(["month"])["total_sales"].sum()

shed.astype('float')

plt.figure(figsize=(14,7))

plt.title('Total Sales Eliminate')

plt.xlabel('Time - (month #)')

plt.ylabel('Sales (# of units)')

plt.plot(shed);
year_sales.head()
x = sales.groupby('city')['total_sales'].agg({'total_sales': sum})

x = x.sort_values(by='total_sales', ascending=False)

x = x.iloc[0:10].reset_index()

x
sales.city.value_counts()
online = year_sales.loc[year_sales.city=='Online']

online.head()
online2 = sales.loc[year_sales.city=='Online']

online2.head()
# Programs decreasing each year

monthlySales = pd.DataFrame(online.groupby(["Month", "year"], as_index=False)["Sales"].sum())



g = sns.FacetGrid(data = monthlySales.sort_values(by="Month"), hue = "year", size = 7, legend_out=True)

g = g.map(plt.plot, "Month", "Sales")

g.add_legend()

plt.subplots_adjust(top=0.9)

g.fig.suptitle('Annual Online Sales', fontsize=22)

g;
on_ts = online2.groupby(["month"])["total_sales"].sum()

ts.astype('float')

plt.figure(figsize=(14,7))

plt.title('Total Sales Online')

plt.xlabel('Time - (month #)')

plt.ylabel('Sales (# of units)')

plt.plot(on_ts);
ts = train.groupby(["month"])["total_sales"].sum()

ts.astype('float')

plt.figure(figsize=(14,7))

plt.title('Total Sales Train Data')

plt.xlabel('Time - (month #)')

plt.ylabel('Sales (# of units)')

plt.plot(ts);
# Assessing the presence of trend & seasonality via Rolling Statistics:



on_line2 = online2.groupby(["month"])["total_sales"].sum()

plt.figure(figsize=(14,7))

plt.plot(on_line2.rolling(window=12,center=False).mean(),label='Rolling Mean - Trend for Online Sales');

plt.plot(on_line2.rolling(window=12,center=False).std(),label='Rolling Standard Deviation - Seasonality for Online Sales');

plt.legend()

plt.show();
# Assessing the presence of trend & seasonality via Rolling Statistics:



plt.figure(figsize=(14,7))

plt.plot(shed.rolling(window=12,center=False).mean(),label='Rolling Mean');

plt.plot(shed.rolling(window=12,center=False).std(),label='Rolling Standard Deviation');

plt.legend()

plt.show();
# Assessing the presence of trend & seasonality via Rolling Statistics:



plt.figure(figsize=(14,7))

plt.plot(keepers.rolling(window=12,center=False).mean(),label='Rolling Mean');

plt.plot(keepers.rolling(window=12,center=False).std(),label='Rolling Standard Deviation');

plt.legend()

plt.show();
# Decomposing Time-Series into Trend, Seasonality, and Residuals

# Is the seasonality Multiplicative or Additive? 

# That is, does the magnitude of the seasonality increase when the time series increases?

# Let's check for Multiplicative Seasonality first with statsmodels.tsa (Time Series Analysis):



import statsmodels.api as sm



ts_decomposed = sm.tsa.seasonal_decompose(on_line2.values,freq=12,model="multiplicative")

plt.figure(figsize=(16,12))

fig = ts_decomposed.plot()

fig.show()
import statsmodels.api as sm

plt.style.use('seaborn-poster')



ts_decomposed = sm.tsa.seasonal_decompose(on_line2.values,freq=12,model="additive")

plt.figure(figsize=(16,12))

fig = ts_decomposed.plot()

fig.show()
plt.style.available
import statsmodels.api as sm



plt.style.use('ggplot')

ts_decomposed = sm.tsa.seasonal_decompose(keepers.values,freq=12,model="multiplicative")

plt.figure(figsize=(16,12))

fig = ts_decomposed.plot()

fig.show()
import statsmodels.api as sm



ts_decomposed = sm.tsa.seasonal_decompose(keepers.values,freq=12,model="additive")

plt.figure(figsize=(16,12))

fig = ts_decomposed.plot()

fig.show()
import statsmodels.api as sm



ts_decomposed = sm.tsa.seasonal_decompose(shed.values,freq=12,model="multiplicative")

plt.figure(figsize=(16,12))

fig = ts_decomposed.plot()

fig.show()
import statsmodels.api as sm



ts_decomposed = sm.tsa.seasonal_decompose(shed.values,freq=12,model="additive")

plt.figure(figsize=(16,12))

fig = ts_decomposed.plot()

fig.show()
# For some reason an additive model is assumed (though, personally, it's hard for me to know why).

# It must be because there is no discernable increase in magnitude each November/December. There is a seasonal increase,

# But that occurs in an additive manner on top of the trend of decreasing sales
# Stationarity tests

def test_stationarity(timeseries):

    

    #Perform Dickey-Fuller test:

    print('Results of Dickey-Fuller Test:')

    dftest = adfuller(timeseries, autolag='AIC')

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

    for key,value in dftest[4].items():

        dfoutput['Critical Value (%s)'%key] = value

    print (dfoutput)



test_stationarity(ts)
# The Null Hypothesis for the Dickey-Fuller Test is that the time-series is not stationary.

# Since the critical value of p=0.05 is not met, the Null Hypothesis cannot be rejected.

# Hence, the data does not yet meet the assumption of stationarity that is required for time-series analysis.
# Now testing the stationarity again after de-seasonality, the Null Hypothesis that the time-series

# Is not stationary can be rejected

test_stationarity(keepers)
# to remove trend

from pandas import Series as Series

# create a differenced series

def difference(dataset, interval=1):

    diff = list()

    for i in range(interval, len(dataset)):

        value = dataset[i] - dataset[i - interval]

        diff.append(value)

    return Series(diff)



# invert differenced forecast

def inverse_difference(last_ob, value):

    return value + last_ob
keep = keep.drop(columns=['category'], inplace=False)

keep.head()
kp1 = keep_1.drop(columns=['category'], inplace=False)

kp1.head()
# # ts=sales.groupby(["date_block_num"])["item_cnt_day"].sum()

# kp.astype('float')

# plt.figure(figsize=(16,16))

# plt.subplot(311)

# plt.title('Original')

# plt.xlabel('Time')

# plt.ylabel('Sales')

# plt.plot(ts)

# plt.subplot(312)

# plt.title('After De-trend')

# plt.xlabel('Time')

# plt.ylabel('Sales')

# new_kp=difference(kp)

# plt.plot(new_kp)

# plt.plot()



# plt.subplot(313)

# plt.title('After De-seasonalization')

# plt.xlabel('Time')

# plt.ylabel('Sales')

# new_kp=difference(kp,12)       # assuming the seasonality is 12 months long

# plt.plot(new_kp)

# plt.plot()
ts_mdl = smt.ARMA(ts.values, order=(1, 1)).fit(method='mle', trend='nc')
print(ts_mdl.summary())
# Figuring out if the time-series is AR (AutoRegressive) or MA (MovingAverage)

# AR [Today = constant + (slope * yesterday) + noise]

# MA [Today = Mean + Noise + (slope * yesterday's noise)]



# We've correctly identified the order of the simulated process as ARMA (p,q), ARMA (1,1):

# p=1 # of lags for AR (AutoRegressive)

# q=1 # of lags for MA (Moving Average)
# adding the dates to the Time-series as index

cs = train.groupby(["month"])["total_sales"].sum()

cs.index = pd.date_range(start = '2013-01-01',end='2015-10-01', freq = 'MS')

cs = cs.reset_index()

cs.head()
# adding the dates to the Time-series as index

keep = keep.groupby(["month"])["total_sales"].sum()

keep.index = pd.date_range(start = '2013-01-01',end='2015-10-01', freq = 'MS')

keep = keep.reset_index()

keep.head()
# kp1 = keep_1.groupby(["month"])["total_sales"].sum()

# kp1.index = pd.date_range(start = '2013-01-01',end='2015-10-01', freq = 'MS')

# kp1 = kp1.reset_index()

# kp1.head()
# adding the dates to the Time-series as index



shed = eliminate.groupby(["month"])["total_sales"].sum()

shed.index = pd.date_range(start = '2013-01-01',end='2015-10-01', freq = 'MS')

shed = shed.reset_index()

shed.head()
# adding the dates to the Time-series as index



on_ts = online2.groupby(["month"])["total_sales"].sum()

on_ts.index = pd.date_range(start = '2013-01-01',end='2015-10-01', freq = 'MS')

on_ts = on_ts.reset_index()

on_ts.head()
from fbprophet import Prophet

#prophet reqiures a pandas df at the below config 

# ( date column named as DS and the value column as Y)



cs.columns = ['ds','y'] # Model 1 Company Sales

keep.columns = ['ds','y'] # Model 2 Retain Sales

shed.columns = ['ds','y'] # Model 3 Relinquish Sales

on_ts.columns = ['ds','y'] # Model 4 Online Sales
model1 = Prophet(yearly_seasonality=True) #instantiate Prophet with only yearly seasonality as our data is monthly 

model1.fit(cs) #fit the model with your dataframe
model2 = Prophet(yearly_seasonality=True) #instantiate Prophet with only yearly seasonality as our data is monthly 

model2.fit(keep) #fit the model with your dataframe
model3 = Prophet(yearly_seasonality=True) #instantiate Prophet with only yearly seasonality as our data is monthly 

model3.fit(shed) #fit the model with your dataframe
model4 = Prophet(yearly_seasonality=True) #instantiate Prophet with only yearly seasonality as our data is monthly 

model4.fit(on_ts) #fit the model with your dataframe
# predict for five months in the future and MS - month start is the frequency

future1 = model1.make_future_dataframe(periods = 12, freq = 'MS')  

# now lets make the forecasts

forecast1 = model1.predict(future1)

forecast1[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
# predict for five months in the future and MS - month start is the frequency

future2 = model2.make_future_dataframe(periods = 12, freq = 'MS')  

# now lets make the forecasts

forecast2 = model2.predict(future2)

forecast2[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
# predict for five months in the future and MS - month start is the frequency

future3 = model3.make_future_dataframe(periods = 12, freq = 'MS')  

# now lets make the forecasts

forecast3 = model3.predict(future3)

forecast3[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
# predict for five months in the future and MS - month start is the frequency

future4 = model4.make_future_dataframe(periods = 12, freq = 'MS')  

# now lets make the forecasts

forecast4 = model4.predict(future4)

forecast4[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
pd.plotting.register_matplotlib_converters()
plt.style.use('fivethirtyeight')



fig1 = model1.plot(forecast1)

plot = plt.suptitle('1C Company 2013-2015 Sales & 2016 Predicted Sales')
fig1 = model1.plot_components(forecast1)
fig2 = model2.plot(forecast2)

plot = plt.suptitle("2013-215 'Retain' Sales & 2016 Predicted 'Retain' Sales")
fig2 = model2.plot_components(forecast2)
fig3 = model3.plot(forecast3)

plot = plt.suptitle("2013-215 'Relinquish' Sales & 2016 Predicted 'Relinquish' Sales")
fig3 = model3.plot_components(forecast3)
fig4 = model4.plot(forecast4)

plot = plt.suptitle("2013-215 Online Sales & 2016 Predicted '2016 Online Sales")
fig4 = model4.plot_components(forecast4)
# Python

from fbprophet.plot import plot_plotly

import plotly.offline as py

py.init_notebook_mode()



fig = plot_plotly(model1, forecast1)  # This returns a plotly Figure

py.iplot(fig)



# 1C COMPANY SALES (Forecasted 12 Months Out)
fig = plot_plotly(model2, forecast2)  # This returns a plotly Figure

py.iplot(fig)



# 'RETAIN' CATEGORIES (Forecasted 12 Months Out)
fig = plot_plotly(model2, forecast2)  # This returns a plotly Figure

py.iplot(fig)



# 'RELINQUISH' CATEGORIES (Forecasted 12 Months Out)
fig = plot_plotly(model2, forecast2)  # This returns a plotly Figure

py.iplot(fig)



# ONLINE SALES (Forecasted 12 Months Out)