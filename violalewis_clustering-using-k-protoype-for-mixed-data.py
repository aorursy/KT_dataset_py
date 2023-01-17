# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn as sns

import matplotlib.pyplot as plt



from datetime import date, datetime



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
def display_all(df):

    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000):

        display(df)
dateparser = lambda x: pd.to_datetime(x, format='%d/%m/%Y', errors='coerce')



data = pd.read_csv("../input/Cluster_fin_csv_ID.csv", parse_dates = ["DOB"], date_parser=dateparser)

display_all(data.head(10)) 
data.rename(columns = {'ID':'code', 'Pincode':'pin', 'Product':'product', 

                       'Main Product':'main_product', 'Inv.Amt':'inv_amt', 'Revenue Amt':'revenue_amt', 'Year':'year',

                       'Occupation':'occupation', 'Annual Income':'annual_income', 'Gender':'gender', 

                       'Marital S':'marital_status', 'DOB':'dob', 'Qualificati':'qualification'},inplace=True)
# Calculate the age from dob



def calculate_age(born):

    today = date.today()

    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))



data['age'] = data['dob'].apply(lambda x : calculate_age(x))
data = data.loc[~(data.dob > date.today())]
data.sort_values('year',ascending=False, inplace= True)

data['count'] = data.groupby(['code', 'main_product'])['code'].transform('count')
fig, axes = plt.subplots(1, 2, figsize = (15,5))

data.groupby('year').size().plot(kind='Bar', color='skyblue', title = 'Record Count per Year', ax=axes[0])

axes[0].set_xlabel = ('Year')

axes[0].set_ylabel = ('Count')



data.groupby('year')['revenue_amt'].sum().plot(kind='Bar', color='skyblue', title = 'Total Revenue by Year', ax=axes[1])

axes[1].set_xlabel = ('Year')

axes[1].set_ylabel = ('Total Revenue')
data = pd.concat([data, pd.get_dummies(data[['year', 'main_product']])],axis=1)

display_all(data.head(10))
data.rename(columns = {'main_product_SD Bond':'main_product_SD_Bond'}, inplace=True)
data.info()
data['total_revenue_amt'] = data.groupby(['code'])['revenue_amt'].transform('sum')
data['year_2012-13'] = data.groupby(['code'])['year_2012-13'].transform('max')

data['year_2013-14'] = data.groupby(['code'])['year_2013-14'].transform('max')

data['year_2014-15'] = data.groupby(['code'])['year_2014-15'].transform('max')

data['year_2015-16'] = data.groupby(['code'])['year_2015-16'].transform('max')

data['year_2016-17'] = data.groupby(['code'])['year_2016-17'].transform('max')

data['year_2017-18'] = data.groupby(['code'])['year_2017-18'].transform('max')

data['year_2018-19'] = data.groupby(['code'])['year_2018-19'].transform('max')
data['main_product_Deposit'] = data.groupby(['code'])['main_product_Deposit'].transform('max')

data['main_product_GI'] = data.groupby(['code'])['main_product_GI'].transform('max')

data['main_product_LI'] = data.groupby(['code'])['main_product_LI'].transform('max')

data['main_product_Locker'] = data.groupby(['code'])['main_product_Locker'].transform('max')

data['main_product_MF'] = data.groupby(['code'])['main_product_MF'].transform('max')

data['main_product_NCD'] = data.groupby(['code'])['main_product_NCD'].transform('max')

data['main_product_SD_Bond'] = data.groupby(['code'])['main_product_SD_Bond'].transform('max')
# Fill null values of age with mean within that group. If still null then fill with mean age for the entire dataset.

data['age'] = data.groupby(['code'])['age'].apply(lambda x: x.fillna(x.mean()))

data['age'] = data['age'].fillna(data['age'].mean())
# Fill null values of dob with 0

data['dob'] = data['dob'].fillna(0)
# Fill na occupation with mode for the column

data['occupation'] = data['occupation'].fillna(data['occupation'].value_counts().index[0])
# Fill null values of qualification with mode within that occupation.

data['qualification'] = data.groupby(['occupation'])['qualification'].apply(lambda x : x.fillna(x.value_counts().index[0]))
# Fill null values of annual income with mode within that occupation.

data['annual_income'] = data.groupby(['occupation'])['annual_income'].apply(lambda x : x.fillna(x.value_counts().index[0]))
# Check if there are any duplicates

data.loc[data.duplicated('code')]
data.drop_duplicates(subset=['code'], keep='first',inplace = True)
# Check if there are any duplicates

data.loc[data.duplicated('code')]
((data.isnull() | data.isna()).sum() * 100/data.index.size).round(2)
data.dropna(how='any', inplace=True)

((data.isnull() | data.isna()).sum() * 100/data.index.size).round(2)
display_all(data.describe(include='all'))
fig, ax = plt.subplots(2,1,figsize=(20,10))

data.age.plot(kind='hist', bins=100, color='skyblue', title = 'Histogram of Age', ax=ax[0])



data.groupby('age')['revenue_amt'].sum().plot(kind='Bar', color='skyblue', title = 'Total Revenue by Age', ax=ax[1])
data.groupby('occupation').size().sort_values(ascending=False).plot(kind='Bar', color='skyblue', title = 'Frequency of Occupation')
data[data.occupation.str.contains('House Wife')].groupby('main_product').size().sort_values(ascending=False).plot(kind='Bar', color='skyblue', title = 'Main Product the house wife invest in')
pd.crosstab(data['main_product'],data['occupation']).plot(kind='bar', figsize=(10,5), title = 'Main Product Vs Occupation')
data.groupby('qualification').size().sort_values(ascending=False).plot(kind='Bar', color='skyblue', title = 'Frequency of Qualification')
data.loc[data['marital_status'].str.contains('MARRIED'), 'marital_status'] = 'Married'

data.loc[data['marital_status'].str.contains('SINGLE'), 'marital_status'] = 'Single'

data.loc[data['marital_status'].str.contains('Separated'), 'marital_status'] = 'Divorced'

data.groupby('marital_status').size().sort_values(ascending=False).plot(kind='Bar', color = 'skyblue', title = 'Frequency of marital status')
# How many values are 'other/others/notanswer' in qualification, occupation and marital_status

data.loc[(data['qualification'] == 'Others') | (data['occupation'] == 'Other') | (data['marital_status'] == 'Notanswer')].count()
# Delete rows 

data = data.loc[~(data['qualification'] == 'Others') & ~(data['occupation'] == 'Other') & ~(data['marital_status'] == 'Notanswer')]



data.loc[(data['qualification'] == 'Others') | (data['occupation'] == 'Other') | (data['marital_status'] == 'Notanswer')].count()
data.pin = data.pin.astype(str)

data = data.loc[~data.pin.str.contains('TRUE|242 00|238 10|200 02|278558?')]

data.pin = data.pin.astype(int)

data = data.loc[data.pin > 110000]

#display_all(data.groupby(['pin']).size())
bins = np.array([100000, 200000, 210000, 220000, 230000, 240000, 250000, 260000, 270000, 280000, 290000, 300000 \

                 , 400000, 500000, 600000 \

                 , 700000, 800000, 900000, 1000000])



labels = pd.cut(data.pin, bins)



grouped = data.groupby(['pin', labels])

grouped.size().unstack(0).T
data['pin_group'] = labels

data.groupby(['pin_group']).size().sort_values(ascending=False).plot(kind='Bar', color = 'skyblue', title = 'Pin group frequency')
data.annual_income = data.annual_income.astype(str)



data.loc[data['annual_income'].str.contains('< 100000'), 'annual_income'] = '50000'

data.loc[data['annual_income'].str.contains('100000 - 200000'), 'annual_income'] = '150000'

data.loc[data['annual_income'].str.contains('200000 - 300000'), 'annual_income'] = '250000'

data.loc[data['annual_income'].str.contains('3.5 LAKHS|300000 - 400000'), 'annual_income'] = '350000'

data.loc[data['annual_income'].str.contains('3.6'), 'annual_income'] = '360000'

data.loc[data['annual_income'].str.contains('100000 - 300000'), 'annual_income'] = '150000'

data.loc[data['annual_income'].str.contains('400000 - 500000'), 'annual_income'] = '450000'

data.loc[data['annual_income'].str.contains('5'), 'annual_income'] = '500000'

data.loc[data['annual_income'].str.contains('5.5|> 500000'), 'annual_income'] = '550000'

data = data.loc[~data.annual_income.str.contains('Not Available')]



data.annual_income = data.annual_income.astype(float)



data = data.loc[data.annual_income > 0]

data.info()
bins = np.array([0, 10000, 100000, 200000, 300000, 400000, 500000, 1000000, 1500000, 10000000])

labels = pd.cut(data.annual_income, bins)



grouped = data.groupby(['main_product', labels])

grouped.size().unstack(0)
data['annual_income_group'] = labels
display_all(data.describe(include='all'))
del data['product']

del data['inv_amt']

del data['year']
data_copy = data.copy()
# Prepare data to feed to the kproto model

columns = ['annual_income_group', 'occupation', 'qualification', 'pin_group', 'gender', 'marital_status'\

          ,'year_2012-13', 'year_2013-14', 'year_2014-15', 'year_2015-16', 'year_2016-17', 'year_2017-18', 'year_2018-19'\

          ,'main_product_Deposit', 'main_product_GI', 'main_product_LI', 'main_product_Locker', 'main_product_MF'\

          ,'main_product_NCD', 'main_product_SD_Bond']



data_cat = data[columns].copy()



for c in columns:

    data_cat[columns] = data_cat[columns].astype('category')



# normalise the continuous variables

data_cat['total_revenue_amt'] = data['total_revenue_amt'].astype(float)

data_cat['total_revenue_amt'] = ((data_cat['total_revenue_amt'] - np.mean(data_cat['total_revenue_amt'])) / np.std(data_cat['total_revenue_amt']))



data_cat['age'] = data['age'].astype(float)

data_cat['age'] = ((data_cat['age'] - np.mean(data_cat['age'])) / np.std(data_cat['age']))



#data_cat['count'] = data['count'].astype(float)

#data_cat['count'] = ((data_cat['count'] - np.mean(data_cat['count'])) / np.std(data_cat['count']))



# Kproto requires input as matrix

data_cat_matrix = data_cat.as_matrix()
data_cat.info()
#Kprototype clustering for data with both numeric and categorical columns

from kmodes import kprototypes



kproto = kprototypes.KPrototypes(n_clusters=5,init='Huang',verbose=1)



clusters = kproto.fit_predict(data_cat_matrix, categorical = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])



# Print the cluster centroids

print(kproto.cluster_centroids_)



#Add cluster column back to original data

data_cl = data.copy()

data_cl['clusters'] = kproto.labels_



#Lets analyze the clusters

print (data_cl.groupby(['clusters']).size())

#display_all(data_cl)
data_cl0 = data_cl.loc[data_cl['clusters'] == 0].describe(include='all')

display_all(data_cl0)
data_cl1 = data_cl.loc[data_cl['clusters'] == 1].describe(include='all')

display_all(data_cl1)
data_cl2 = data_cl.loc[data_cl['clusters'] == 2].describe(include='all')

display_all(data_cl2)
data_cl3 = data_cl.loc[data_cl['clusters'] == 3].describe(include='all')

display_all(data_cl3)
data_cl4 = data_cl.loc[data_cl['clusters'] == 4].describe(include='all')

display_all(data_cl4)
fig, ax = plt.subplots()

data_cl.groupby(['clusters']).size().plot(kind='Bar', color='skyblue', ax=ax, title ="Cluster Size", figsize=(7, 5), fontsize=12)

ax.set_xlabel('Clusters')

ax.set_ylabel('Count')

fig.savefig('img1.png')
fig, ax = plt.subplots()

data_cl.groupby(['clusters'])['total_revenue_amt'].sum().plot(kind='Bar', color='skyblue', ax=ax, title ="Total Revenue per Cluster", figsize=(7, 5), fontsize=12)

ax.set_xlabel('Clusters')

ax.set_ylabel('Total Revenue')

fig.savefig('img2.png')
fig, ax = plt.subplots()

data_cl.groupby(['clusters'])['total_revenue_amt'].mean().plot(kind='Bar', color='skyblue', ax=ax, title ="Mean Revenue per Cluster", figsize=(7, 5), fontsize=12)

ax.set_xlabel('Clusters')

ax.set_ylabel('Mean Revenue')

fig.savefig('img3.png')
fig, ax = plt.subplots()

data_cl.groupby(['clusters'])['age'].mean().plot(kind='Bar', color='skyblue', ax=ax, title ="Mean Age per Cluster", figsize=(7, 5), fontsize=12)

ax.set_xlabel('Clusters')

ax.set_ylabel('Mean Age')

fig.savefig('img4.png')
data_cl.sort_values(['clusters', 'total_revenue_amt'], ascending = [True, False], inplace=True)
writer = pd.ExcelWriter('fin_clusters.xlsx')

data_cl.to_excel(writer,'Clusters', index=False)



data_cl0 = data_cl.loc[data_cl['clusters'] == 0].describe(include='all')

data_cl0.to_excel(writer,'Cluster_info', startcol=12,startrow=1)



workbook  = writer.book

worksheet = writer.sheets['Cluster_info']



worksheet.write(0, 12, 'Cluster 0 Information')



worksheet.write(14, 12, 'Cluster 1 Information')

data_cl1.to_excel(writer, sheet_name = 'Cluster_info', startcol=12,startrow=15)



worksheet.write(29, 12, 'Cluster 2 Information')

data_cl2.to_excel(writer, sheet_name = 'Cluster_info', startcol=12,startrow=30)



worksheet.write(44, 12, 'Cluster 3 Information')

data_cl3.to_excel(writer, sheet_name = 'Cluster_info', startcol=12,startrow=45)



worksheet.write(59, 12, 'Cluster 4 Information')

data_cl4.to_excel(writer, sheet_name = 'Cluster_info', startcol=12,startrow=60)



worksheet.insert_image('A1', 'img1.png')

worksheet.insert_image('A25', 'img2.png')

worksheet.insert_image('A50', 'img3.png')

worksheet.insert_image('A75', 'img4.png')



writer.save()