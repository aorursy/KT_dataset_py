# for linear algebra

import numpy as np 



# for data processing, CSV file I/O (e.g. pd.read_csv)

import pandas as pd 



# for visualizations

import matplotlib.pyplot as plt

import seaborn as sns
# Importing Available files

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# MEMORY REDUCTION FOR BETTER PERFORMANCE

def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
%time d1 = pd.read_csv("/kaggle/input/global-food-prices/wfp_market_food_prices.csv",encoding="ISO-8859-1")

print(d1.shape)
d1.info
# improving performance by reducing memory usage to work comfortably on the data.



reduce_mem_usage(d1)
# Number of rows for each country



country_unique, country_freq = np.unique(d1['adm0_name'], return_counts = True)

listrows = []

for i in range(country_unique.shape[0]):

    

    print(country_unique[i], ': ', country_freq[i])

    listrows.append( [country_unique[i], country_freq[i]])
listrows
df = pd.DataFrame(listrows, columns = ['Country', 'RowCount'])

df= df.sort_values(by=['RowCount'],ascending=False)

ax= df.plot(kind='bar', y = 'RowCount',x ='Country',    

    legend = False,figsize=(64,32), fontsize=24)

ax.set_xlabel("Country",fontsize=24)

ax.set_ylabel("RowCount",fontsize=24)

plt.show()
turkey = d1.loc[d1['adm0_name'] == 'Turkey', 'cur_name']

turkey_unique = np.unique(turkey)

print(turkey)
%time cur = pd.read_excel("/kaggle/input/currency/currency.xlsx")
cur.info
cur['USD per Unit_Avg']


d1.rename(columns={'adm0_id':'country_id',

                   'adm0_name':'country',

                   'adm1_id':'province_id',

                   'adm1_name':'province',

                   'mkt_id':'city_id',

                   'mkt_name':'city',

                   'cm_id':'food_id',

                   'cm_name':'food',

                   'mp_month':'month',

                   'mp_year':'year',

                   'mp_price':'price',

                   'mp_commoditysource':'source',

                   'um_name':'unit',

                   'cur_name':'currency',

                   'cur_id':'currency_id',

                   'um_id':'unit_id',

                   'pt_name':'purchase_type',

                   'pt_id':'purchase_type_id'},

                   inplace=True)
# getting unique currencies



currency_ = d1['currency'].unique()

currency_
# turkey, iran, syria, iraq

f1= d1[d1['country'] == 'Turkey'].head(1) 

f2= d1[d1['country'] == 'Iran'].head(1)

f3= d1[d1['country'] == 'Iraq'].head(1)

f4= d1[d1['country'] == 'Syria'].head(1)

result = pd.concat([f1,f2,f3,f4])

result
# Looking for Syria if it exists in the data

d1[d1['country'].str.contains('Syria')]
# Looking for Iran if it exists in the data

d1[d1['country'].str.contains('Iran')]
# Rename our filters as correct names for countries

f1= d1[d1['country'] == 'Turkey'].head(1) 

f2= d1[d1['country'].str.contains('Iran')].head(1)

f3= d1[d1['country'] == 'Iraq'].head(1)

f4= d1[d1['country'].str.contains('Syria')].head(1)

result = pd.concat([f1,f2,f3,f4])

result
result['price']


currency_try = cur[cur['Currency_Code'] == 'TRY']

currency_try = currency_try['USD per Unit_Avg']

currency_try
currency_irr = cur[cur['Currency_Code'] == 'IRR']

currency_irr = currency_irr['USD per Unit_Avg']

currency_irr
currency_iqd = cur[cur['Currency_Code'] == 'IQD']

currency_iqd = currency_iqd['USD per Unit_Avg']

currency_iqd
currency_syp = cur[cur['Currency_Code'] == 'SYP']

currency_syp = currency_syp['USD per Unit_Avg']
# correlation matrix for the attributes of the data (it is meaningless for categorical data. 

#I am leaving it like this in need of a further use )



#corrmat = d1.corr()

#mask = np.zeros_like(corrmat, dtype=np.bool)

#mask[np.triu_indices_from(mask)] = True



#f, ax = plt.subplots(figsize=(12, 9))

#sns.heatmap(corrmat,mask=mask, vmax=1, square=True);
d1.describe()
product_types = d1['food'].unique()

count = 0

for i in product_types:

     count+=1

count
# Calculating the average price of all products

items =pd.DataFrame(d1.groupby("food")["price"].agg("mean").sort_values(ascending=False))[:321]

items.head(5)
d1.columns # Columns of the data.
d1.head() # Getting the first 5 rows of the data.
d1.tail() # Getting the first 5 rows of the data.
country = d1['country'].unique()

count = 0

for i in country: 

    count += 1

    

print("Unique Country Number:",count)



food = d1['food'].unique()

count = 0

for i in food: 

    count += 1

    

print("Unique Food Number:",count)



source = d1['source'].unique()

count = 0

for i in source: 

    count += 1

    

print("Unique Source Number:",count)



purchase_type = d1['purchase_type'].unique()

count = 0

for i in purchase_type: 

    count += 1

    

print("Purchase Type Number:",count)
# Unique City Count for each Country



dups_color_and_shape = d1.pivot_table(index=['country','city'], aggfunc='size')

dups = d1.drop_duplicates(['country', 'city'], keep='last')

city_count = dups.groupby('country').count()

#city_count

a = city_count.sort_values('country_id',ascending=False)

a.reset_index()

a['country_id']

plt.style.use("ggplot")



plt.figure(figsize=(8,10))

plt.gcf().subplots_adjust(left=.3)

fig = sns.barplot(x=a.country_id,y=a.index,data=a)

plt.gca().set_title("City Count For Each Country")

fig.set(xlabel='City Count', ylabel='Country')

plt.show()
#Turkey

turkey1 = d1.loc[d1['country'] == 'Turkey' , ['country','city','purchase_type','food','unit','price','month','year']]

turkey1
#Iran

iran1 = d1.loc[d1['country'].str.contains('Iran') , ['country','city','purchase_type','food','unit','price','month','year']]

iran1
#Iraq

iraq1 = d1.loc[d1['country'].str.contains('Iraq') , ['country','city','purchase_type','food','unit','price','month','year']]

iraq1
#Syria

syria1 = d1.loc[d1['country'].str.contains('Syria') , ['country','city','purchase_type','food','unit','price','month','year']]

syria1
turkey = turkey1.copy()

turkey['price'] = turkey['price'] * float(currency_try)

turkey


iran = iran1.copy()

iran['price'] = iran['price'] * float(currency_irr)

iran


iraq = iraq1.copy()

iraq['price'] = iraq['price'] * float(currency_iqd)

iraq
syria = syria1.copy()

syria['price'] = syria['price'] * float(currency_syp)

syria
turkey_top10=pd.DataFrame(turkey.groupby("food")["price","unit"].agg("mean").sort_values(by="price",ascending=False))[:10]

turkey_top = turkey.copy()

turkey_top.sort_values("food", inplace = True) 

turkey_top.drop_duplicates(subset = "food", 

                     keep = 'first', inplace = True)

turkey_top10
turkey.loc[turkey['food'] == 'Fuel (gas)', ['price']] = turkey.loc[turkey['food'] == 'Fuel (gas)', ['price']] / 12
turkey[turkey['food'].isin(['Fuel (gas)'])]
plt.style.use("ggplot")

turkey_top10 = turkey.copy()

turkey_top10=pd.DataFrame(turkey.groupby("food")["price"].agg("mean").sort_values(ascending=False))[:10]

plt.figure(figsize=(8,10))

plt.gcf().subplots_adjust(left=.3)

sns.barplot(x=turkey_top10.price,y=turkey_top10.index,data=turkey_top10)

plt.gca().set_title("Top 10 items produced in Turkey")

plt.show()
# Taking array of unique products for each country

tu = turkey['food'].unique()

ira = iran['food'].unique()

irq = iraq['food'].unique()

syr = syria['food'].unique()



# Joining arrays to identify these products

all_countries = np.concatenate([tu,ira,irq,syr])



# printing count of products to spot common products

food_unique, food_frequency = np.unique(all_countries, return_counts = True)



for i in range(food_unique.shape[0]):

    

    print(food_unique[i], ': ', food_frequency[i])

# Finding which countries owns these products



turkey_temp = turkey[turkey['food'].isin(['Wheat flour', 'Sugar','Rice','Lentils','Yogurt','Fuel (gas)', 'Tomatoes'])]

turkey_temp = turkey_temp.copy()

turkey_temp.sort_values("food", inplace = True) 

turkey_temp.drop_duplicates(subset = "food", 

                     keep = 'first', inplace = True)

turkey_temp
iran_temp = iran[iran['food'].isin(['Wheat flour', 'Sugar','Rice','Lentils','Yogurt','Fuel (gas)', 'Tomatoes'])]

iran_temp = iran_temp.copy()

iran_temp.sort_values("food", inplace = True) 

iran_temp.drop_duplicates(subset ="food", 

                     keep = 'first', inplace = True)

iran_temp
iraq_temp = iraq[iraq['food'].isin(['Wheat flour', 'Sugar','Rice','Lentils','Yogurt','Fuel (gas)', 'Tomatoes'])]

iraq_temp = iraq_temp.copy()

iraq_temp.sort_values("food", inplace = True) 

iraq_temp.drop_duplicates(subset ="food", 

                     keep = 'first', inplace = True)

iraq_temp
syria_temp = syria[syria['food'].isin(['Wheat flour', 'Sugar','Rice','Lentils','Yogurt','Fuel (gas)', 'Tomatoes'])]

syria_temp = syria_temp.copy()

syria_temp.sort_values("food", inplace = True) 

syria_temp.drop_duplicates(subset ="food", 

                     keep = 'first', inplace = True)

syria_temp
t3 = d1[(d1['country'] == 'Turkey')]

t3['city'].unique()
#sns.distplot(a=d1["price"],rug=True)
plt.figure(figsize=(12,5))

plt.title("Distribution Price of Rice in Turkey")

turkey_rice = turkey[turkey['food'].isin(['Rice'])]

ax = sns.distplot(turkey_rice["price"], color = 'r')



plt.figure(figsize=(12,5))

plt.title("Distribution Price of Rice in Iraq")

iraq_rice = iraq[iraq['food'].isin(['Rice'])]

ax = sns.distplot(iraq_rice["price"], color = 'b')



plt.figure(figsize=(12,5))

plt.title("Distribution Price of Rice in Syria")

syria_rice = syria[syria['food'].isin(['Rice'])]

ax = sns.distplot(syria_rice["price"], color = 'y')
plt.figure(figsize=(12,5))

plt.title("Distribution Price of Lentils in Turkey")

turkey_lentils = turkey[turkey['food'].isin(['Lentils'])]

ax = sns.distplot(turkey_lentils["price"], color = 'r')



plt.figure(figsize=(12,5))

plt.title("Distribution Price of Lentils in Iran")

iran_lentils = iran[iran['food'].isin(['Lentils'])]

ax = sns.distplot(iran_lentils["price"], color = 'g')



plt.figure(figsize=(12,5))

plt.title("Distribution Price of Lentils in Syria")

syria_lentils = syria[syria['food'].isin(['Lentils'])]

ax = sns.distplot(syria_lentils["price"], color = 'y')
plt.figure(figsize=(12,5))

plt.title("Distribution Price of Sugar in Turkey")

turkey_sugar = turkey[turkey['food'].isin(['Sugar'])]

ax = sns.distplot(turkey_sugar["price"], color = 'r')



plt.figure(figsize=(12,5))

plt.title("Distribution Price of Sugar in Iraq")

iran_sugar = iran[iran['food'].isin(['Sugar'])]

ax = sns.distplot(iran_sugar["price"], color = 'g')



plt.figure(figsize=(12,5))

plt.title("Distribution Price of Sugar in Syria")

syria_sugar = syria[syria['food'].isin(['Sugar'])]

ax = sns.distplot(syria_sugar["price"], color = 'y')



plt.figure(figsize=(12,5))

plt.title("Distribution Price of Sugar in Syria")

iraq_sugar = iraq[iraq['food'].isin(['Sugar'])]

ax = sns.distplot(iraq_sugar["price"], color = 'b')
syria_lentils.describe()
syria_sugar.describe()
syria_rice.describe()
turkey_lentils.describe()
turkey_sugar.describe()
turkey_rice.describe()
q1 = syria_lentils['price'].quantile(.25)

print('Lentils'' q1 is {}'.format(q1))

q3 = syria_lentils['price'].quantile(.75)

print('Lentils'' q3 is {}'.format(q3))

IQR = q3- q1

syria_lentils_v2 = syria_lentils.copy()

syria_lentils_v2.drop(syria_lentils_v2[(syria_lentils_v2['price'] < 0) | (syria_lentils_v2['price'] > (1.5 * IQR)) ].index , inplace=True)

print(syria_lentils_v2['price'].value_counts())
q1 = syria_sugar['price'].quantile(.25)

print('Sugar'' q1 is {}'.format(q1))

q3 = syria_lentils['price'].quantile(.75)

print('Sugar'' q3 is {}'.format(q3))

IQR = q3- q1

syria_sugar_v2 = syria_sugar.copy()

syria_sugar_v2.drop(syria_sugar_v2[(syria_sugar_v2['price'] < 0) | (syria_sugar_v2['price'] > (1.5 * IQR)) ].index , inplace=True)

print(syria_sugar_v2['price'].value_counts())
q1 = syria_rice['price'].quantile(.25)

print('Rice'' q1 is {}'.format(q1))

q3 = syria_rice['price'].quantile(.75)

print('Rice'' q3 is {}'.format(q3))

IQR = q3- q1

syria_rice_v2 = syria_lentils.copy()

syria_rice_v2.drop(syria_rice_v2[(syria_rice_v2['price'] < 0) | (syria_rice_v2['price'] > (1.5 * IQR)) ].index , inplace=True)

print(syria_rice_v2['price'].value_counts())
syria_lentils_v2.describe()
syria_sugar_v2.describe()
syria_rice_v2.describe()
plt.figure(figsize=(12,5))

plt.title("Distribution Price of Lentils in Syria")

ax = sns.distplot(syria_lentils_v2["price"], color = 'y')
plt.figure(figsize=(12,5))

plt.title("Distribution Price of Sugar in Syria")

ax = sns.distplot(syria_sugar_v2["price"], color = 'y')
plt.figure(figsize=(12,5))

plt.title("Distribution Price of Rice in Syria")

ax = sns.distplot(syria_rice_v2["price"], color = 'y')
syria_lentils['price'].skew()
syria_lentils_v2['price'].skew()
syria_sugar['price'].skew()
syria_sugar_v2['price'].skew()
syria_rice['price'].skew()
syria_rice_v2['price'].skew()