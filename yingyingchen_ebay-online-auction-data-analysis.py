%matplotlib inline 



import pandas as pd             # data package

import matplotlib.pyplot as plt # graphics 

import seaborn as sns           # graphics 

import datetime as dt           # date tools, used to note current date  

import numpy as ny

from scipy import stats
#read csv file 

auction=pd.read_csv("auction.csv")

auction.head()
auction.shape
auction.head()
auction4=auction[['auctionid','bid','price','auction_type']]
auction4.head()
closing_price = auction4.groupby(['auctionid']).last()
closing_price.head()
closing_price['check']=np.where((closing_price['bid'] == closing_price['price']), closing_price['bid'], np.nan)
closing_price.isnull().sum()
closing_price.shape
1-146/628
auction1 = auction[['auctionid','bidtime']]
auction1.head()
max_bidtime = auction1.groupby(['auctionid']).max()
max_bidtime.head()
fig, ax = plt.subplots(figsize = (18, 16))

ax.hist(max_bidtime['bidtime'], bins=25, normed=True)

sns.kdeplot(max_bidtime['bidtime'], ax=ax, lw = 3)

ax.set_title("Histogram of bidtime")

plt.show()
auction.head()
num_type=auction.groupby('auction_type')['auctionid'].nunique()

num_type=num_type.to_frame()

print(num_type)
# Create a list of colors (from iWantHue)

colors=['#5ABA10', '#FE110E','#CA5C05']

# Create a pie chart

plt.pie(

    # using data total)arrests

    num_type['auctionid'],

    # with the labels being officer names

    labels=num_type.index,

    # with no shadows

    shadow=False,

    # with colors

   colors=colors,

    # with the percent listed as a fraction

    autopct='%1.1f%%',

    )



# View the plot drop above

plt.axis('equal')



# View the plot

plt.tight_layout()

plt.show()
max_bidtime=auction.groupby('auctionid',as_index=False)['bidtime'].max()
max_bidtime.head()
auction2 = auction[['auctionid','auction_type']]
auction_type=auction2.groupby('auctionid',as_index=False)['auction_type'].first()
auction_type.head()
auction3 = pd.merge(max_bidtime, auction_type, how='inner', on=['auctionid'])
auction3.head()
seven_day_auction = auction3.query('auction_type == "7 day auction"') 

five_day_auction= auction3.query('auction_type == "5 day auction"') 

three_day_auction= auction3.query('auction_type == "3 day auction"') 
print(three_day_auction.shape,five_day_auction.shape,seven_day_auction.shape)
fig, ax = plt.subplots(figsize = (8, 6))

ax.hist(seven_day_auction['bidtime'], bins=25, normed=True,label='seven_day')

ax.hist(three_day_auction['bidtime'], bins=25, normed=True,label='three_day')

ax.hist(five_day_auction['bidtime'], bins=25, normed=True,label='five_day')

ax.set_title("Histogram of max bidtime for different auction types")

plt.legend()

plt.show()
auction.head()
open_bid = auction.groupby(['auctionid']).last()

open_bid=open_bid[['openbid','item','auction_type']]

open_bid.head()
#create a boxplot first in order to have a more direct way of seeing means of open bid for different types

sns.set_style("whitegrid")

ax = sns.boxplot(x="item", y="openbid", data=open_bid)
df=open_bid[open_bid['item']=='Cartier wristwatch']['openbid']

df.max()
#drop the row that has openbid=5000.0

value_list = [5000.0]

open_bid1=open_bid[~open_bid.openbid.isin(value_list)]

open_bid1.head()
fig, ax = plt.subplots(figsize = (14, 13))

sns.distplot(open_bid1[open_bid['item']=='Cartier wristwatch']['openbid'], kde=True,label="Cartier wristwatch")

sns.distplot(open_bid1[open_bid['item']=='Palm Pilot M515 PDA']['openbid'], kde=True,label="Palm Pilot M515 PDA")

sns.distplot(open_bid1[open_bid['item']=='Xbox game console']['openbid'], kde=True,label='Xbox game console')

plt.legend()
fig, ax = plt.subplots(figsize = (8, 4.5))

ax = sns.boxplot(x="item", y="openbid", data=open_bid1)
fig, ax = plt.subplots(figsize = (8, 4.5))

ax = sns.boxplot(x="item", y="openbid", data=open_bid1)

ax.set(ylim=(-20, 1000))
open_bid1.groupby('item')['openbid'].median()
open_bid1.groupby('item')['openbid'].mean()
open_bid1.groupby('item')['openbid'].var()
# compute one-way ANOVA P value   

from scipy import stats  

watch=open_bid1[open_bid1['item']=='Cartier wristwatch']['openbid']

PDA=open_bid1[open_bid1['item']=='Palm Pilot M515 PDA']['openbid']

console=open_bid1[open_bid1['item']=='Xbox game console']['openbid']



f_val, p_val = stats.f_oneway(watch, PDA, console)  

  

print("One-way ANOVA P =", p_val)
seven_days=open_bid1[open_bid1['auction_type']=='7 day auction']['openbid']

five_days=open_bid1[open_bid1['auction_type']=='5 day auction']['openbid']

three_days=open_bid1[open_bid1['auction_type']=='3 day auction']['openbid']



f_val, p_val = stats.f_oneway(seven_days, five_days, three_days)  

  

print("One-way ANOVA P =", p_val)
pct_change=auction[['auctionid','openbid','price','item','auction_type']].groupby(['auctionid']).last()

pct_change.head()
pct_change.groupby('item')['price'].median()
pct_change.groupby('item')['price'].mean()
pct_change.groupby('item')['price'].var()
auction.head()
num_bids=auction.groupby('auctionid')['bidder'].count()

num_bids=num_bids.to_frame()

num_bids.columns=['num_bids']

num_bids.head()
#open_bid1 is a dataframe that drops the extreme openbid value of 5,000

open_bid1.head()
num_bids=pd.merge(num_bids, open_bid1, left_index=True, right_index=True)
num_bids.head()
sns.jointplot(x="num_bids", y="openbid", data=num_bids,kind='reg')
num_bids.groupby('item').corr()
num_bids.groupby('auction_type').corr()
num_bids.groupby(['auction_type','item']).corr()
num_bids.head()
#number of auctions for each auction type 

num_bids.groupby('auction_type')['num_bids'].count()
#total number of bids for each auction type 

num_bids.groupby('auction_type')['num_bids'].sum()
#averge number of bids for each type of auctions 

num_bids.groupby('auction_type')['num_bids'].sum()/num_bids.groupby('auction_type')['num_bids'].count()
ave_bids=num_bids.groupby(['auction_type','item'])['num_bids'].sum()/num_bids.groupby(['auction_type','item'])['num_bids'].count()
ave_bids=ave_bids.to_frame()
ave_bids
ave_bids1=ave_bids.reset_index(level=['auction_type','item'])

ave_bids1
from bokeh.charts import Bar, output_file, show

from bokeh.sampledata.autompg import autompg as df
#plot a bar chart for grouped data

p=Bar(ave_bids1, label="auction_type", values="num_bids", group="item", legend="top_left", ylabel='average_bids')

show(p)
ave_bids1.groupby(['item'])['num_bids'].var()