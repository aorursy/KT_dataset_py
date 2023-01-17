import numpy as np
import pandas as pd
import os


import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns
import plotly
plotly.tools.set_credentials_file(username='vvr', api_key='rWFSioMoYeB5dnZJzz8k')
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls


import squarify
import sklearn
df = pd.read_csv('../input/kiva_loans.csv')
df.describe(include= 'all').T
df.info()
df2  =pd.read_csv('../input/kiva_mpi_region_locations.csv')
#df2.describe(include = 'all').T
df2.sample(4)
df2.describe(include = 'all').T
df3 = pd.read_csv('../input/loan_theme_ids.csv')
df3.describe(include ='all').T
df4 = pd.read_csv('../input/loan_themes_by_region.csv')
df4.describe(include = 'all').T
#df.funded_amount.value_counts()
#df.funded_amount.sort_values()
df.funded_amount.mean()
#np.log(df[df.funded_amount>0].dropna().diff())
#df.funded_amount
afteroutlayers =  df[(df.funded_amount>0) & (df.funded_amount <6000)]
afteroutlayers2 = df[(df.loan_amount>0) & (df.loan_amount <6000)]
afteroutlayers3 = df[(df.lender_count>0) & (df.lender_count <300)]
#afteroutlayers.head()

#sns.barplot(df.funded_amount.value_counts[:20])
color = sns.color_palette("husl", 8)
f, (ax0, ax1) = plt.subplots(1,2, figsize = (16,5))

#plt.figure(figsize = (10,6))
'''sns.countplot(x = 'funded_amount', data = df[df["funded_amount"].isin(df["funded_amount"].value_counts()[:20].index)],
                        order=df["funded_amount"].value_counts().iloc[:20].index, ax= ax0)
'''
sns.distplot(afteroutlayers.funded_amount, kde = False, ax = ax0)
ax0.set_title('Peak Values of funded amount')
ax0.set_xlabel('Funded amount')
ax0.set_ylabel('Frequency')
#ax.set_xticks( rotation = 90)

sns.distplot(afteroutlayers2.loan_amount, kde = False, ax = ax1)
ax1.set_title('Peak Values of loan amount')
ax1.set_xlabel('Loan amount')
ax1.set_ylabel('Frequency')
plt.figure(figsize = (10,5))
sns.countplot(y= 'sector', data = df[df["sector"].isin(df["sector"].value_counts()[:30].index)],
                        order=df["sector"].value_counts().iloc[:30].index)
#plt.xticks(rotation = 90)
plt.ylabel('Sector')
plt.xlabel('Frequency')
plt.title('Peak Values of the Sector')
#plt.y
plt.figure(figsize = (10,5))
sns.countplot(y= 'activity', data = df[df["activity"].isin(df["activity"].value_counts()[:20].index)],
                        order=df["activity"].value_counts().iloc[:20].index)
#plt.xticks(rotation = 90)
plt.ylabel('Activity')
plt.xlabel('Count')
plt.title('Peak Values of the Activity')
plt.figure(figsize = (14,8))
sns.countplot(y= 'use', data = df[df["use"].isin(df["use"].value_counts()[:20].index)],
                        order=df["use"].value_counts().iloc[:20].index)
#plt.xticks(rotation = 90)
plt.ylabel('use')
plt.xlabel('Count')
plt.title('Peak Values of the use')
#plt.figure(figsize = (10,7))

f, (ax0, ax1) = plt.subplots(1,2, figsize = (18,8))
sns.countplot(y= 'country', data = df[df["country"].isin(df["country"].value_counts()[:30].index)],
                        order=df["country"].value_counts().iloc[:30].index, ax = ax0)
#plt.xticks(rotation = 90)
ax0.set_ylabel('country')
ax0.set_xlabel('Frequency')
ax0.set_title('Frequency of the Counties')


sns.countplot(y= 'currency', data = df[df["currency"].isin(df["currency"].value_counts()[:30].index)],
                        order=df["currency"].value_counts().iloc[:30].index, ax = ax1)
#ax1.xticks(rotation = 90)
ax1.set_ylabel('currency')
ax1.set_xlabel('Count')
ax1.set_title('Peak Values of the currency')
#inspired from SRK
cou = pd.DataFrame(df['country'].value_counts()).reset_index()
cou.columns = ['country', 'loans']
cou = cou.reset_index().drop('index', axis=1)
#print(cou)
import plotly.offline as offline
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

data = [ dict(
        type = 'choropleth',
        locations = cou['country'],
        z = cou['loans'],
        text = cou['country'],
        colorscale = [[0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],\
            [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"]],
        autocolorscale = False,
        reversescale = True,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            tickprefix = '',
            title = 'Total Number of loans'),
      ) ]

layout = dict(
    title = 'Loans per country',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict( data=data, layout=layout )
py.iplot( fig, validate=False, filename='loans_world_map' )
plt.figure(figsize = (10,5))
sns.countplot(x= 'region', data = df[df["region"].isin(df["region"].value_counts()[:30].index)],
                        order=df["region"].value_counts().iloc[:30].index)
plt.xticks(rotation = 90)
plt.xlabel('region')
plt.ylabel('Count')
plt.title('Peak Values of the region')
'''
#monyer = df.groupby([df.monyer.dt.year, df.monyer.dt.month]).count()['id'].unstack()
#sns.pointplot(monyer)
df['posted_time'] = pd.to_datetime(df['posted_time'])
df['disbursed_time'] = pd.to_datetime(df['disbursed_time'])
df['funded_time'] = pd.to_datetime(df['funded_time'])
df['date'] = pd.to_datetime(df['date'])

yearp = df.posted_time.dt.year
yeard = df.disbursed_time.dt.year
yearf = df.funded_time.dt.year
yearda = df.date.dt.year

yearsump = yearp.value_counts()
yearsumd = yeard.value_counts()
yearsumf = yearf.value_counts()
yearsumda = yeard.value_counts()

#f, (ax0, ax1, ax2,ax3) = plt.subplots(4, figsize = (18,10))
f, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2,figsize = (18,12))
yearsump.plot( ax = ax0)
ax0.set_title('Posted Time')
ax0.set_xlabel('Time')
ax0.set_ylabel('Frequency')

yearsump.plot( ax = ax1)
ax1.set_title('Distributed Time')
ax1.set_xlabel('Time')
ax1.set_ylabel('Frequency')

yearsumf.plot( ax = ax2)
#sns.stripplot(yearsumf, ax=ax2)
ax2.set_title('Funded Time')
ax2.set_xlabel('Time')
ax2.set_ylabel('Frequency')

yearsumda.plot( ax = ax3)
#sns.stripplot(yearsumf, ax=ax2)
ax3.set_title('Date')
ax3.set_xlabel('Date')
ax3.set_ylabel('Frequency')
'''
'''
yearsump.plot()
yearsumd.plot()
yearsumf.plot()
yearsumda.plot()
'''
#df['term_in_months'].value_counts(sort = False)
#datasor = df.sort_values(df.term_in_months, axis = 1)
#df.sort_values(by = ['term_in_months'], axis=1)
datasor = df.sort_values('term_in_months')
#datasor.sample(3)
#plt.figure(figsize=(8,6))

'''df.groupby(df['term_in_months'].value_counts(), [pd.cut(df['term_in_months'], np.arange(0,100,10))])\
       .size()\
       .unstack(0)\
       .plot.bar(stacked=True)
       
'''

#datasor['term_in_months'].value_counts(sort = False)
#datasor['term_in_months'].head(30)
f, (ax0, ax1, ax2) = plt.subplots(1,3,figsize = (20, 6))

#datasor['term_in_months'].plot(kind = 'hist', ax = ax0)
sns.distplot(df.term_in_months, kde = False, ax = ax0)
ax0.set_xlabel('Term in Months')
ax0.set_ylabel('Frequency')
ax0.set_title('Term in Months')


sns.distplot(afteroutlayers3.lender_count, kde = False, ax = ax1)
ax1.set_xlabel('Lender Counts')
ax1.set_ylabel('Frequency')
ax1.set_title('Lender Counts')

sns.countplot(df.repayment_interval, ax = ax2)
ax2.set_xlabel('Repayment Interval')
ax2.set_ylabel('Frequency')
ax2.set_title('Repayment Interval')
df['borrower_genders'] = df['borrower_genders'].str.lower()
df.borrower_genders.str.replace('[^A-Za-z\s]+,', '').str.split(expand=True)
df['borrower_genders'] = df['borrower_genders'].str.replace(',', '')
df['tags'] = df['tags'].str.lower()
df.tags.str.replace('[^A-Za-z\s]+,', '').str.split(expand=True)
df['tags'] = df['tags'].str.replace(',', '')
df['tags'] = df['tags'].str.replace('#', '')

f, (ax0, ax1) = plt.subplots(1,2, figsize = (22,6))

df.borrower_genders.str.split(expand=True).stack().value_counts().plot(kind = 'bar', ax=ax0)
ax0.set_xlabel('Gender')
ax0.set_ylabel('Frequency')
ax0.set_title('Gender difference')


df.tags.str.split(expand=True).stack().value_counts().plot(y = 'tags',kind = 'bar', ax=ax1)
ax1.set_xlabel('Tags')
ax1.set_ylabel('Frequency')
ax1.set_title('Tags')



'''
import nltk
from stop_words import get_stop_words
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from textblob import TextBlob


stop = stopwords.words('english')
#TextBlob(df['use'][1]).words
token = word_tokenize(df.use[1])
#print(x)
#x = [1,2,3,4]
#print(df['use'])
df['use'].apply(lambda x: [item for item in x if item not in stop])
df['use_without_stopwords'] = df['use'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

'''
plt.figure(figsize = (30, 40))
sns.set(style="ticks")
sns.boxplot(x='funded_amount', y = 'country', data = df)
sns.despine(offset=10, trim=True)
plt.title('Amount funded for each Country')
plt.figure(figsize = (20, 10))
sns.set(style="ticks")
sns.boxplot(x = 'funded_amount', y = 'sector', data = df)
sns.despine(offset=10, trim=True)
plt.title('Amount funded across each Sector')
#plt.figure(figsize = (10,10))
f, (ax0, ax1) = plt.subplots(1,2, figsize =(22, 10)) 
sns.countplot(y= 'ISO', data = df2[df2["ISO"].isin(df2["ISO"].value_counts()[:30].index)],
                        order=df2["ISO"].value_counts().iloc[:30].index,ax = ax1)
#plt.xticks(rotation = 90)
ax1.set_ylabel('ISO')
ax1.set_xlabel('Frequency')
ax1.set_title('Frequency of ISO')

sns.countplot(y= 'country', data = df2[df2["country"].isin(df2["country"].value_counts()[:30].index)],
                        order=df2["country"].value_counts().iloc[:30].index, ax =ax0)
#plt.xticks(rotation = 90)
ax0.set_ylabel('country')
ax0.set_xlabel('Frequency')
ax0.set_title('Frequency of country')
data = df2[['MPI','country','region']].copy()
data.dropna(inplace = True)
data.sort_values(['MPI','country']).head(30)
data.sort_values(['MPI','country']).tail(40)
df3.columns
plt.figure(figsize = (10,5))
sns.countplot(y= 'Loan Theme Type', data = df3[df3["Loan Theme Type"].isin(df3["Loan Theme Type"].value_counts()[:20].index)],
                        order=df3["Loan Theme Type"].value_counts().iloc[:20].index)

plt.ylabel('Loan Theme Type')
plt.xlabel('Frequency')
plt.title('Loan Theme Type')
plt.figure(figsize = (10,5))
sns.distplot(df3['Partner ID'].dropna(), kde = False)
#plt.xticks(rotation = 90)
plt.xlabel('Partner ID')
plt.ylabel('Frequency')
plt.title('Distribution of Partner ID')
df3.columns
'''
plt.figure(figsize = (20, 50))
sns.set(style="ticks")
sns.boxplot(x='Partner ID', y = 'Loan Theme Type', data = df3)
sns.despine(offset=10, trim=True)
plt.title('Amount funded for e')
'''

'''plt.figure(figsize = (10,5))
sns.barplot(y= 'Loan Theme ID', data = df3[df3["Loan Theme ID"].isin(df3["Loan Theme ID"].value_counts()[:20].index)],
                        order=df3["Loan Theme ID"].value_counts().iloc[:20].index)
#plt.xticks(rotation = 90)
plt.ylabel('Loan Theme ID')
plt.xlabel('Frequency')
plt.title('Loan Theme ID')
'''
plt.figure(figsize = (10,8))
sns.countplot(y= 'Field Partner Name', 
                     data = df4[df4["Field Partner Name"].isin(df4["Field Partner Name"].value_counts()[:30].index)],
                        order=df4["Field Partner Name"].value_counts().iloc[:30].index)
#plt.xticks(rotation = 90)
plt.ylabel('Field Partner Name')
plt.xlabel('Frequency')
plt.title('Field Partner Name')
#plt.figure(figsize = (10,8))
f,(ax0,ax1) = plt.subplots(1,2, figsize = (18, 6))
sns.countplot(y= 'sector', data = df4[df4["sector"].isin(df4["sector"].value_counts()[:30].index)],
                        order=df4["sector"].value_counts().iloc[:30].index, ax= ax0 )
ax0.set_ylabel('Sector')
ax0.set_xlabel('Frequency')
ax0.set_title('Sector')

sns.countplot(x = 'forkiva', data = df4, ax = ax1)
ax1.set_ylabel('Frequency')
ax1.set_xlabel('For Kiva')
ax1.set_title('Is it For KIVA')
df4.columns
dataforkiva = df4[df4.amount<4000]
plt.figure(figsize = (30, 8))
sns.set(style="ticks")
sns.boxplot(x='forkiva', y = 'amount', data = dataforkiva, hue ='sector')
sns.despine(offset=10, trim=True)
plt.title('Sector vs FORKIVA')
lis = ['Philippines','Kenya','El Salvador','Cambodia','Pakistan','Peru','Colombia','Uganda']
dfkiva = df4[df4.country.isin(lis)]
dfkiva.head()
dataforkiva2 = dfkiva[dfkiva["amount"]<4000]
plt.figure(figsize = (30, 8))
sns.set(style="ticks")
sns.boxplot(x='forkiva', y = 'amount', data = dataforkiva2, hue = dataforkiva2.country)
sns.despine(offset=10, trim=True)
plt.title('Amount funded for each Country')
df['country'].value_counts().head(3)
df_phi = df[df.country == 'Philippines']
df_phi2 = df2[df2.country == 'Philippines']
df_phi4 = df4[df4.country == 'Philippines']
#df_phi.count()
afteroutlayers_phi =  df_phi[(df_phi.funded_amount>0) & (df_phi.funded_amount <1500)]
afteroutlayers_phi2 = df_phi[(df_phi.loan_amount>0) & (df_phi.loan_amount <1500)]
afteroutlayers_phi3 = df_phi[(df_phi.lender_count>0) & (df_phi.lender_count <300)]

f, (ax0, ax1) = plt.subplots(1,2, figsize = (16,5))

#plt.figure(figsize = (10,6))
'''sns.countplot(x = 'funded_amount', data = df[df["funded_amount"].isin(df["funded_amount"].value_counts()[:20].index)],
                        order=df["funded_amount"].value_counts().iloc[:20].index, ax= ax0)
'''
sns.distplot(afteroutlayers_phi.funded_amount, kde = False, ax = ax0)
ax0.set_title('Peak Values of funded amount for Philippines')
ax0.set_xlabel('Funded amount ')
ax0.set_ylabel('Frequency')
#ax.set_xticks( rotation = 90)

sns.distplot(afteroutlayers_phi2.loan_amount, kde = False, ax = ax1)
ax1.set_title('Peak Values of loan amount for Philippines')
ax1.set_xlabel('Loan amount')
ax1.set_ylabel('Frequency')
f, (ax0, ax1) = plt.subplots(1,2, figsize =(24,7))
sns.countplot(y= 'sector', data = df_phi[df_phi["sector"].isin(df_phi["sector"].value_counts()[:20].index)],
                        order=df_phi["sector"].value_counts().iloc[:30].index, ax = ax1)
#plt.xticks(rotation = 90)
ax1.set_ylabel('Sector')
ax1.set_xlabel('Count')
ax1.set_title('Peak Values of the Sector for Philippines ')

sns.countplot(y= 'activity', data = df_phi[df_phi["activity"].isin(df_phi["activity"].value_counts()[:20].index)],
                        order=df_phi["activity"].value_counts().iloc[:20].index, ax = ax0)
ax0.set_ylabel('Activity')
ax0.set_xlabel('Count')
ax0.set_title('Peak Values of the Activity for Philippines')
 #monyer = df.groupby([df.monyer.dt.year, df.monyer.dt.month]).count()['id'].unstack()
#sns.pointplot(monyer)
'''
df['posted_time'] = pd.to_datetime(df['posted_time'])
df['disbursed_time'] = pd.to_datetime(df['disbursed_time'])
df['funded_time'] = pd.to_datetime(df['funded_time'])
df['date'] = pd.to_datetime(df['date'])

yearp_phi  = df_phi.posted_time.dt.year
yeard_phi  = df_phi.disbursed_time.dt.year
yearf_phi  = df_phi.funded_time.dt.year
yearda_phi = df_phi.date.dt.year

yearsump_phi = yearp_phi.value_counts()
yearsumd_phi = yeard_phi.value_counts()
yearsumf_phi = yearf_phi.value_counts()
yearsumda_phi = yeard_phi.value_counts()

#f, (ax0, ax1, ax2,ax3) = plt.subplots(4, figsize = (18,10))
f, ((ax0, ax1)) = plt.subplots(1, 2,figsize = (18,6))
yearsump_phi.plot( ax = ax0)
ax0.set_title('Posted Time')
ax0.set_xlabel('Time')
ax0.set_ylabel('Frequency')

yearsumd_phi.plot( ax = ax1)
ax1.set_title('Distributed Time')
ax1.set_xlabel('Time')
ax1.set_ylabel('Frequency')


yearsumf.plot( ax = ax2)
#sns.stripplot(yearsumf, ax=ax2)
ax2.set_title('Funded Time')
ax2.set_xlabel('Time')
ax2.set_ylabel('Frequency')

yearsumda.plot( ax = ax3)
#sns.stripplot(yearsumf, ax=ax2)
ax3.set_title('Date')
ax3.set_xlabel('Date')
ax3.set_ylabel('Frequency')

'''

data2 = df_phi2[['MPI','region']].copy()
data2.dropna(inplace = True)
data2.sort_values(['MPI','region']).head(30)
f, (ax0,ax1) = plt.subplots(1,2, figsize = (15,6))

sns.countplot(x = 'forkiva', data = df_phi4, ax = ax0)
ax0.set_ylabel('Frequency')
ax0.set_xlabel('For Kiva')
ax0.set_title('Is it For KIVA')

dataforkiva2 = df_phi4[df_phi4.amount<4000]
#plt.set_set(style="ticks")
sns.boxplot(x='forkiva', y = 'amount', data = dataforkiva2, hue ='sector', ax = ax1)
ax1.set_title('Sector vs FORKIVA')


df_k = df[df.country == 'Kenya']
df_k2 = df2[df2.country == 'Kenya']
df_k4 = df4[df4.country == 'Kenya']
#df_phi.count()
afteroutlayers_k =  df_k[(df_k.funded_amount>0) & (df_k.funded_amount <1500)]
afteroutlayers_k2 = df_k[(df_k.loan_amount>0) & (df_k.loan_amount <1500)]
afteroutlayers_k3 = df_k[(df_k.lender_count>0) & (df_k.lender_count <300)]

f, (ax0, ax1) = plt.subplots(1,2, figsize = (16,5))

#plt.figure(figsize = (10,6))
'''sns.countplot(x = 'funded_amount', data = df[df["funded_amount"].isin(df["funded_amount"].value_counts()[:20].index)],
                        order=df["funded_amount"].value_counts().iloc[:20].index, ax= ax0)
'''
sns.distplot(afteroutlayers_k.funded_amount, kde = False, ax = ax0)
ax0.set_title('Peak Values of funded amount for Kenya')
ax0.set_xlabel('Funded amount ')
ax0.set_ylabel('Frequency')
#ax.set_xticks( rotation = 90)

sns.distplot(afteroutlayers_k2.loan_amount, kde = False, ax = ax1)
ax1.set_title('Peak Values of loan amount for Kenya')
ax1.set_xlabel('Loan amount')
ax1.set_ylabel('Frequency')
f, (ax0, ax1) = plt.subplots(1,2, figsize =(24,7))
sns.countplot(y= 'sector', data = df_k[df_k["sector"].isin(df_k["sector"].value_counts()[:20].index)],
                        order=df_k["sector"].value_counts().iloc[:30].index, ax = ax1)
#plt.xticks(rotation = 90)
ax1.set_ylabel('Sector')
ax1.set_xlabel('Count')
ax1.set_title('Peak Values of the Sector for Kenya ')

sns.countplot(y= 'activity', data = df_k[df_k["activity"].isin(df_k["activity"].value_counts()[:20].index)],
                        order=df_k["activity"].value_counts().iloc[:20].index, ax = ax0)
ax0.set_ylabel('Activity')
ax0.set_xlabel('Count')
ax0.set_title('Peak Values of the Activity for Kenya')
data2 = df_k2[['MPI','region']].copy()
data2.dropna(inplace = True)
data2.sort_values(['MPI','region'])
f, (ax0,ax1) = plt.subplots(2, figsize = (10,10))

sns.countplot(x = 'forkiva', data = df_k4, ax = ax0)
ax0.set_ylabel('Frequency')
ax0.set_xlabel('For Kiva')
ax0.set_title('Is it For KIVA')

dataforkiva2 = df_k4[df_k4.amount<4000]
#plt.set_set(style="ticks")
sns.boxplot(x='forkiva', y = 'amount', data = dataforkiva2, hue ='sector', ax = ax1)
ax1.set_title('Sector vs FORKIVA')


df_i =   df[df.country == 'India']
df_i2 = df2[df2.country == 'India']
df_i4 = df4[df4.country == 'India']
#df_phi.count()
afteroutlayers_i =  df_i[(df_i.funded_amount>0) & (df_i.funded_amount <1500)]
afteroutlayers_i2 = df_i[(df_i.loan_amount>0) & (df_i.loan_amount <1500)]
afteroutlayers_i3 = df_i[(df_i.lender_count>0) & (df_i.lender_count <300)]
f, (ax0, ax1) = plt.subplots(1,2, figsize = (16,5))

#plt.figure(figsize = (10,6))
'''sns.countplot(x = 'funded_amount', data = df[df["funded_amount"].isin(df["funded_amount"].value_counts()[:20].index)],
                        order=df["funded_amount"].value_counts().iloc[:20].index, ax= ax0)
'''
sns.distplot(afteroutlayers_i.funded_amount, kde = False, ax = ax0)
ax0.set_title('Peak Values of funded amount for INDIA')
ax0.set_xlabel('Funded amount ')
ax0.set_ylabel('Frequency')
#ax.set_xticks( rotation = 90)

sns.distplot(afteroutlayers_i2.loan_amount, kde = False, ax = ax1)
ax1.set_title('Peak Values of loan amount for INDIA')
ax1.set_xlabel('Loan amount')
ax1.set_ylabel('Frequency')
f, (ax0, ax1) = plt.subplots(1,2, figsize =(24,7))
sns.countplot(y= 'sector', data = df_i[df_i["sector"].isin(df_i["sector"].value_counts()[:20].index)],
                        order=df_i["sector"].value_counts().iloc[:30].index, ax = ax1)
#plt.xticks(rotation = 90)
ax1.set_ylabel('Sector')
ax1.set_xlabel('Count')
ax1.set_title('Peak Values of the Sector for India')

sns.countplot(y= 'activity', data = df_i[df_i["activity"].isin(df_i["activity"].value_counts()[:20].index)],
                        order=df_i["activity"].value_counts().iloc[:20].index, ax = ax0)
ax0.set_ylabel('Activity')
ax0.set_xlabel('Count')
ax0.set_title('Peak Values of the Activity for India')
f, (ax0,ax1) = plt.subplots(2, figsize = (16,10))

sns.countplot(x = 'forkiva', data = df_i4, ax = ax0)
ax0.set_ylabel('Frequency')
ax0.set_xlabel('For Kiva')
ax0.set_title('Is it For KIVA')

dataforkiva3 = df_i4[df_i4.amount<4000]
#plt.set_set(style="ticks")
sns.boxplot(x='forkiva', y = 'amount', data = dataforkiva3, hue ='sector', ax = ax1)
ax1.set_title('Sector vs FORKIVA')

