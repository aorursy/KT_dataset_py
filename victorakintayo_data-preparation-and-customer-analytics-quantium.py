import numpy as np

import pandas as pd

import datetime as dt

import re

from nltk.util import ngrams



import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib.dates as mdates



from mlxtend.frequent_patterns import apriori

from mlxtend.frequent_patterns import association_rules
Transaction_Data = pd.read_csv('../input/transaction/QVI_transaction_data.csv')

Customer_Data = pd.read_csv('../input/customer-data/QVI_purchase_behaviour.csv')

Transaction_Data.head()
Transaction_Data.dtypes
Transaction_Data.describe()
Customer_Data.head()


Transaction_Data.DATE = pd.TimedeltaIndex(Transaction_Data.DATE, unit='d') + dt.datetime(1899, 12, 30)

Transaction_Data
# get unique products

Transaction_Data['PROD_NAME'].unique()
# Extracting product size

Transaction_Data['PACK_SIZE'] = [re.search(r"[0-9]+(g|G)", i).group(0).replace('G','g') for i in Transaction_Data['PROD_NAME']]



Transaction_Data.PROD_NAME = Transaction_Data.PROD_NAME.replace('\d+g', "")

Transaction_Data
# replace & with space and remove multiple spaces

Transaction_Data.PROD_NAME = [" ".join(i.replace('&',' ').split()) for i in Transaction_Data.PROD_NAME]

# remove digits that are followed by grams

Transaction_Data.PROD_NAME = [re.sub(r"\s*[0-9]+(g|G)", r"", i) for i in Transaction_Data.PROD_NAME]

Transaction_Data
def replaceWords(string):

    # specific

    string = re.sub(r"SeaSalt", "Sea Salt", string)

    string = re.sub(r"Frch/Onin", "French Onion", string)

    string = re.sub(r"Cheddr Mstrd", "Cheddar Mustard", string)

    string = re.sub(r"Jlpno Chili", "Jalapeno Chilli", string)

    string = re.sub(r"Swt/Chlli Sr/Cream", "Sweet Chilli Sour Cream", string)

    string = re.sub(r"SourCream", "Sour Cream", string)

    string = re.sub(r"Tmato Hrb Spce", "Tomato Herb Spice", string)

    string = re.sub(r"S/Cream", "Sour Cream", string)

    string = re.sub(r"ChipsFeta", "Chips Feta", string)

    string = re.sub(r"ChpsHny", "Chips Honey", string)

    string = re.sub(r"FriedChicken", "Fried Chicken", string)

    string = re.sub(r"OnionDip", "Onion Dip", string)

    string = re.sub(r"SweetChili", "Sweet Chilli", string)

    string = re.sub(r"PotatoMix", "Potato Mix", string)

    string = re.sub(r"Seasonedchicken", "Seasoned Chicken", string)

    string = re.sub(r"CutSalt/Vinegr", "Cut Salt Vinegar", string)

    string = re.sub(r"ChpsBtroot", "Chips Beetroot", string)

    string = re.sub(r"ChipsBeetroot", "Chips Beetroot", string)

    string = re.sub(r"ChpsFeta", "Chips Feta", string)

    string = re.sub(r"OnionStacked", "Onion Stacked", string)

    string = re.sub(r"Ched", "Cheddar", string)

    string = re.sub(r"Strws", "Straws", string)

    string = re.sub(r"Slt", "Salt", string)

    string = re.sub(r"Chikn", "Chicken", string)

    string = re.sub(r"Rst", "Roast", string)

    string = re.sub(r"Vinegr", "Vinegar", string)

    string = re.sub(r"Mzzrlla", "Mozzarella", string)

    string = re.sub(r"Originl", "Original", string)

    string = re.sub(r"saltd", "Salted", string)

    string = re.sub(r"Swt", "Sweet", string)

    string = re.sub(r"Chli", "Chilli", string)

    string = re.sub(r"Hony", "Honey", string)

    string = re.sub(r"Chckn", "Chicken", string)

    string = re.sub(r"Chp", "Chips", string)

    string = re.sub(r"Chip", "Chips", string)

    string = re.sub(r"Btroot", "Beetroot", string)

    string = re.sub(r"Chs", "Cheese", string)

    string = re.sub(r"Crm", "Cream", string)

    string = re.sub(r"Orgnl", "Original", string)

    string = re.sub(r"Swt ChliS/Cream", "Sweet Chilli Sour Cream", string)

    string = re.sub(r"SnagSauce", "Snag Sauce", string)

    string = re.sub(r"Compny", "Company", string)

    string = re.sub(r"HoneyJalapeno", "Honey Jalapeno", string)

    string = re.sub(r"Sweetspcy", "Sweet Spicy", string)

    string = re.sub(r"BeetrootRicotta", "Beetroot Ricotta", string)

    string = re.sub(r"Crn", "Corn", string)

    string = re.sub(r"Crnchers", "Crunchers", string)

    string = re.sub(r"CreamHerbs", "CreamHerbs", string)

    string = re.sub(r"Tmato", "Tomato", string)

    string = re.sub(r"BBQMaple", "Berbeque Maple", string)

    string = re.sub(r"BBQ", "Berbeque", string)





    return string



Transaction_Data['PROD_NAME'] = [replaceWords(s) for s in Transaction_Data['PROD_NAME']]



Transaction_Data['PROD_NAME'].replace('Infzns Crn Crnchers Tangy Gcamole',

'Infuzions Corn Crunchers Tangy Guacamole', inplace=True)
# Removing special characters, replace & with space

Transaction_Data['PROD_NAME']=Transaction_Data['PROD_NAME'].replace('\&','',regex=True)
#Remove Salsa Products

#Transaction_Data.drop([Transaction_Data.PROD_NAME == '\Salsa'], inplace = True)

Transaction_Data = Transaction_Data[~Transaction_Data['PROD_NAME'].str.contains('Salsa')]

#Transaction_Data Summary

Transaction_Data.describe()
Transaction_Data[Transaction_Data['PROD_QTY'] > 5]
Customer_Data = Customer_Data[Customer_Data['LYLTY_CARD_NBR'] != 226000]

Customer_Data
#Remove Outliers

#Transaction_Data Summary

Transaction_Data = Transaction_Data[Transaction_Data['PROD_QTY'] < 200]

Transaction_Data.describe()
#Transaction By Date



Transaction_Count_By_Date = Transaction_Data.groupby('DATE').agg({'PROD_QTY': 'sum'}).reset_index()



Transaction_Count_By_Date
Transaction_Data.append({'DATE' : '2018-12-25'} , ignore_index=True)
import datetime

start_date = '2018-07-01'

end_date   = '2019-06-30'



All_Dates = pd.date_range(start_date, end_date).tolist()

All_Dates = pd.Series(All_Dates, name='NEW_DATE')

All_Dates
Transaction_Count_By_Date=Transaction_Count_By_Date.merge(All_Dates, how='outer', left_index=True, right_index=True)

Transaction_Count_By_Date.tail(10)
Missing = Transaction_Count_By_Date[(Transaction_Count_By_Date['DATE'] >= '2018-12-01') & (Transaction_Count_By_Date['DATE'] <= '2018-12-31')]

Missing
December = Transaction_Data[(Transaction_Data['DATE'] >= '2018-12-01') & (Transaction_Data['DATE'] <= '2018-12-31')]

December
Transactions_in_December = December.groupby('DATE').agg({'PROD_QTY': 'sum'}).reset_index()

Transactions_in_December
## Plot December quantities sold

# filter december

Transactions_in_December = Transaction_Count_By_Date[Transaction_Count_By_Date['DATE'].isin(pd.date_range(start="2018-12-01",end="2018-12-31").tolist())]

Transaction_Count_By_Date

# fill in missing dec data

#Transaction_Count_By_Date = Transaction_Count_By_Date.set_index('DATE').reindex(pd.date_range(start="2018-12-01",end="2018-12-31"), fill_value=0)

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(20,5))



ax1=plt.subplot(121)

sns.lineplot(x="DATE", y="PROD_QTY", data=Transaction_Count_By_Date, ax=ax1)

ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b-%Y"))

sns.lineplot(x="DATE", y="PROD_QTY", data=Transaction_Count_By_Date[(Transaction_Count_By_Date['DATE'] > '2018-12-11') & (Transaction_Count_By_Date['DATE'] < '2018-12-28')], color='#2ca02c', ax=ax1)

sns.lineplot(x="DATE", y="PROD_QTY", data=Transaction_Count_By_Date[(Transaction_Count_By_Date['DATE'] > '2018-08-12') & (Transaction_Count_By_Date['DATE'] < '2018-08-24')], color='red', ax=ax1)

sns.lineplot(x="DATE", y="PROD_QTY", data=Transaction_Count_By_Date[(Transaction_Count_By_Date['DATE'] > '2019-05-10') & (Transaction_Count_By_Date['DATE'] < '2019-05-24')], color='red', ax=ax1)

plt.ylabel('Quantities Sold')

plt.title('Quantities Sold Throughout Whole Year')



## Plot December quantities sold

# filter december

QTY_December = Transaction_Count_By_Date[Transaction_Count_By_Date['DATE'].isin(pd.date_range(start="2018-12-01",end="2018-12-31").tolist())]



# fill in missing dec data

QTY_December = QTY_December.set_index('DATE').reindex(pd.date_range(start="2018-12-01",end="2018-12-31"), fill_value=0)



ax2=plt.subplot(122)

ax2.bar(QTY_December.index,QTY_December['PROD_QTY'],color='#2ca02c')

ax2.set_xticks(QTY_December.index)

ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b-%d"))

ax2.xaxis.set_minor_formatter(mdates.DateFormatter("%b-%d"))

ax2.tick_params(axis='x', rotation=90) 

plt.ylabel('Quantities Sold')

plt.title('Quantities Sold in December')

plt.show()
Transaction_Data.PACK_SIZE
# Product Size



fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(15,4))



ax1=plt.subplot(121)

Transaction_Data.groupby('PACK_SIZE').agg({'PROD_QTY': 'sum'}).sort_values('PROD_QTY').reset_index().plot.barh(x='PACK_SIZE', legend=False, ax=ax1)

ax1.set_ylabel('Pack Size')

ax1.set_xlabel('Quantities Sold')



plt.show()




# get brand name from first word

Transaction_Data['BRAND_NAME'] = [i.split(' ')[0] for i in Transaction_Data['PROD_NAME']]

Transaction_Data
Transaction_Data.BRAND_NAME.unique()
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,4))



# Product quantity sales by brand

ax1=plt.subplot(121)

Transaction_Data.groupby(['BRAND_NAME'], as_index=False).agg({'PROD_QTY': 'sum'}).sort_values('PROD_QTY').plot.barh(x='BRAND_NAME',legend=False, ax=ax1)



ax1.set_xlabel('Quantity Sold')

ax1.set_ylabel('BRAND_NAME')

ax1.set_title('Quantities Sold by Brand')



ax2=plt.subplot(122)

Transaction_Data.groupby(['BRAND_NAME'], as_index=False)[['TXN_ID']].count().sort_values('TXN_ID').plot.barh(x='BRAND_NAME',color='#ff7f0e', legend=False, ax=ax2)

ax2.set_xlabel('Number of Transactions')

ax2.set_ylabel('BRAND_NAME')

ax2.set_title('Transactions by Brand')



plt.show()
Customer_Data
#Analysis of customer class

Customer_Class = pd.pivot_table(data=Customer_Data[['LIFESTAGE','PREMIUM_CUSTOMER']],index=['PREMIUM_CUSTOMER'], aggfunc=np.size)

Customer_Class
sns.set()

Customer_Class.plot(kind='barh', alpha=.9, color=sns.color_palette("colorblind"),title='Customer Class Analysis').invert_yaxis()

plt.ylabel("Customer Class")
# Customer lifestage counts

Customer_Data.LIFESTAGE.value_counts().plot(kind='barh', alpha=.9, color=sns.color_palette("colorblind"), title='Customer Lifestage Analysis').invert_yaxis()
# Merge Customer Data to Transaction Data using LYLTY_CARD as Primary Key

Merged_Data = Transaction_Data.merge(Customer_Data, on='LYLTY_CARD_NBR')



#export merged data as csv

Merged_Data.to_csv(r'./Data.csv', index = False)

# check for duplicates

print('No Duplicates:', len(Merged_Data) == len(Merged_Data)) 

# check for nulls

print('Number of Nulls:', Merged_Data.isnull().sum().sum()) 
Merged_Data.head(10)
# Sum up for Lifestage and Premium_Customer group 

Premium_Lifestyle = Merged_Data.groupby(['LIFESTAGE','PREMIUM_CUSTOMER']).agg({'TOT_SALES':'sum','PROD_QTY':'sum', 'TXN_ID':'count'}).reset_index().sort_values('TOT_SALES', ascending = False) 



Premium_Lifestyle

pd.pivot_table(Merged_Data, values='TOT_SALES', index=['LIFESTAGE'], columns=['PREMIUM_CUSTOMER'], aggfunc=np.sum, margins=True,margins_name='Total')

pd.pivot_table(Merged_Data, values='TOT_SALES', index=['LIFESTAGE'], columns=['PREMIUM_CUSTOMER'], aggfunc=np.sum).plot( kind='barh')


pd.pivot_table(Merged_Data, values='PROD_QTY', index=['LIFESTAGE'], columns=['PREMIUM_CUSTOMER'], aggfunc=np.size, margins=True,margins_name='Total')

pd.pivot_table(Merged_Data, values='PROD_QTY', index=['LIFESTAGE'], columns=['PREMIUM_CUSTOMER'], aggfunc=np.size).plot( kind='barh')
# Number of unique customers in each group

Premium_Lifestyle_Customers = Merged_Data[['LYLTY_CARD_NBR','LIFESTAGE','PREMIUM_CUSTOMER']].drop_duplicates('LYLTY_CARD_NBR').reset_index(drop=True).groupby(['LIFESTAGE','PREMIUM_CUSTOMER']).size().reset_index(name='Count').sort_values('Count').merge(Premium_Lifestyle, on=['LIFESTAGE','PREMIUM_CUSTOMER'])





Premium_Lifestyle_Customers['SALES_PER_Customer'] = Premium_Lifestyle_Customers['TOT_SALES']/Premium_Lifestyle_Customers['TXN_ID']

Premium_Lifestyle_Customers['SALES_PER_Unique_Customer'] = Premium_Lifestyle_Customers['TOT_SALES']/Premium_Lifestyle_Customers['Count']

Premium_Lifestyle_Customers = Premium_Lifestyle_Customers.sort_values('SALES_PER_Customer')



Premium_Lifestyle_Customers
## Sales by each Segment per Customer and per Unique Customer

#Sales per customer

pd.pivot_table(Premium_Lifestyle_Customers, values='SALES_PER_Customer', index=['LIFESTAGE'], columns=['PREMIUM_CUSTOMER'], aggfunc=np.sum).plot(kind='barh')

#Sales per Unique customer

pd.pivot_table(Premium_Lifestyle_Customers, values='SALES_PER_Unique_Customer', index=['LIFESTAGE'], columns=['PREMIUM_CUSTOMER'], aggfunc=np.sum).plot(kind='barh')
Premium_Lifestyle_Customers['QTY_PER_CUSTOMER'] =Premium_Lifestyle_Customers['PROD_QTY']/Premium_Lifestyle_Customers['TXN_ID']

Premium_Lifestyle_Customers['QTY_PER_UNIQUE_CUSTOMER'] = Premium_Lifestyle_Customers['PROD_QTY']/Premium_Lifestyle_Customers['Count']

Premium_Lifestyle_Customers = Premium_Lifestyle_Customers.sort_values('QTY_PER_CUSTOMER')

Premium_Lifestyle_Customers
## Sales Per Customer

pd.pivot_table(Premium_Lifestyle_Customers, values='QTY_PER_CUSTOMER', index=['LIFESTAGE'], columns=['PREMIUM_CUSTOMER'], aggfunc=np.sum).plot(kind='barh')
## Sales Per Unique Customer

pd.pivot_table(Premium_Lifestyle_Customers, values='QTY_PER_UNIQUE_CUSTOMER', index=['LIFESTAGE'], columns=['PREMIUM_CUSTOMER'], aggfunc=np.sum).plot(kind='barh')
#Price per unit

Merged_Data['PRICE_PER_UNIT'] = Merged_Data['TOT_SALES']/Merged_Data['PROD_QTY'] 

# get price per unit of each customer then groupby lifestage and premium_customer to get average per group

price_per_unit = Merged_Data.groupby('LYLTY_CARD_NBR').agg({'PRICE_PER_UNIT':'mean'}).reset_index().merge(Merged_Data[['LYLTY_CARD_NBR','LIFESTAGE','PREMIUM_CUSTOMER']], on='LYLTY_CARD_NBR').groupby(['LIFESTAGE','PREMIUM_CUSTOMER']).agg({'PRICE_PER_UNIT':'mean'}).reset_index().sort_values('PRICE_PER_UNIT')

price_per_unit
## Price per segment

pd.pivot_table(price_per_unit, values='PRICE_PER_UNIT', index=['LIFESTAGE'], columns=['PREMIUM_CUSTOMER'], aggfunc=np.sum).plot(kind='barh')
basket = (Merged_Data[(Merged_Data['LIFESTAGE']=='YOUNG SINGLES/COUPLES') & (Merged_Data['PREMIUM_CUSTOMER']=='Mainstream')]

        .groupby(['LYLTY_CARD_NBR','BRAND_NAME'])['PROD_QTY']

        .sum().unstack().reset_index().fillna(0)

        .set_index('LYLTY_CARD_NBR'))



def encode_units(x):

    if x <= 0:

        return 0

    if x >= 1:

        return 1



basket_sets = basket.applymap(encode_units)

basket_sets
frequent_itemsets = apriori(basket_sets, min_support=0.07, use_colnames=True)

frequent_itemsets
rules = association_rules(frequent_itemsets, metric="lift")

rules.head()
young_mainstream = Merged_Data[(Merged_Data['LIFESTAGE']=='YOUNG SINGLES/COUPLES') & (Merged_Data['PREMIUM_CUSTOMER']=='Mainstream')]

quantity_bybrand = young_mainstream.groupby(['BRAND_NAME'])[['PROD_QTY']].sum().reset_index()

quantity_bybrand.PROD_QTY = quantity_bybrand.PROD_QTY / young_mainstream.PROD_QTY.sum()

quantity_bybrand = quantity_bybrand.rename(columns={"PROD_QTY": "Targeted_Segment"})



other_segments = pd.concat([Merged_Data, young_mainstream]).drop_duplicates(keep=False) 



# remove young_mainsream

quantity_bybrand_other = other_segments.groupby(['BRAND_NAME'])[['PROD_QTY']].sum().reset_index()

quantity_bybrand_other.PROD_QTY = quantity_bybrand_other.PROD_QTY / other_segments.PROD_QTY.sum()

quantity_bybrand_other = quantity_bybrand_other.rename(columns={"PROD_QTY": "Other_Segment"})



quantity_bybrand = quantity_bybrand.merge(quantity_bybrand_other, on ='BRAND_NAME')

quantity_bybrand['Affinitytobrand'] = quantity_bybrand['Targeted_Segment'] / quantity_bybrand['Other_Segment']

quantity_bybrand = quantity_bybrand.sort_values('Affinitytobrand')



quantity_bybrand.head()
Merged_Data




quantity_bysize = young_mainstream.groupby(['PACK_SIZE'])[['PROD_QTY']].sum().reset_index()

quantity_bysize.PROD_QTY = quantity_bysize.PROD_QTY / young_mainstream.PROD_QTY.sum()

quantity_bysize = quantity_bysize.rename(columns={"PROD_QTY": "Targeted_Segment"})



quantity_bysize_other = other_segments.groupby(['PACK_SIZE'])[['PROD_QTY']].sum().reset_index()

quantity_bysize_other.PROD_QTY = quantity_bysize_other.PROD_QTY / other_segments.PROD_QTY.sum()

quantity_bysize_other = quantity_bysize_other.rename(columns={"PROD_QTY": "Other_Segment"})



quantity_bysize = quantity_bysize.merge(quantity_bysize_other, on='PACK_SIZE')

quantity_bysize['Affinitytosize'] = quantity_bysize['Targeted_Segment'] / quantity_bysize['Other_Segment']

quantity_bysize = quantity_bysize.sort_values('Affinitytosize')
# Function for Lollipop chart

def loll_plot(df1,x,y,xlabel,title,firstX):

    

    my_color=np.where(df1[x]==firstX, '#ff7f0e', '#1f77b4')

    my_color[0] = 'red'

    my_size=np.where(df1[x]==firstX, 70, 30)

    my_size[0] = '70'



    plt.hlines(y=np.arange(0,len(df1)),xmin=0,xmax=df1[y],color=my_color)

    plt.scatter(df1[y], np.arange(0,len(df1)), color=my_color, s=my_size)

    plt.yticks(np.arange(0,len(df1)), df1[x])

    plt.xlabel(xlabel)

    plt.title(title)
fig = plt.figure(figsize=(10,6))

ax1 = plt.subplot(121)

loll_plot(quantity_bybrand,'BRAND_NAME','Affinitytobrand','Affinity','Affinity To Brand','Tyrrells')



ax2 = plt.subplot(122)

loll_plot(quantity_bysize,'PACK_SIZE','Affinitytosize','Affinity','Affinity To Product Size','270')



plt.suptitle('Young Singles/Couples Mainstream')

plt.show()