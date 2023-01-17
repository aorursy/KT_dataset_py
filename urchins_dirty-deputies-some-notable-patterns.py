# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



%matplotlib inline

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sb

import re

import datetime



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



#from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/dirty_deputies_v2.csv', na_values='Nan', dtype={'party_tse': str, 'party_nmembers':str, 'party_ideology2':str, 'party_ideology3':str, 'party_ideology4':str})
##Plot column names

df.columns
##Examine variable levels##

print(df.head())
## What can we say about the political parties?

dfParty = pd.DataFrame()

dfParty['ndeputies'] = partyDeputies= df.groupby(['political_party'])['deputy_name'].nunique()

dfParty['ntyperefunds'] = partyRefundType = df.groupby(['political_party'])['refund_description'].nunique()

dfParty['nstates'] = partyStates = df.groupby(['political_party'])['deputy_state'].nunique()

dfParty['ncompanies'] = partyCompanies = df.groupby(['political_party'])['company_name'].nunique()

dfParty['nrefunds'] = partyNumberRefunds = df.groupby(['political_party'])['refund_value'].count()

dfParty['nrefundsdeputy'] = partyNumberRefunds / partyDeputies

dfParty['totalrefundamount'] = partyTotalAmount = df.groupby(['political_party'])['refund_value'].sum()

dfParty['averagerefundamount'] = partyAverageAmount = df.groupby(['political_party'])['refund_value'].mean()

dfParty['totalperdeputy']= partyTotalAmountPolitician = partyTotalAmount / partyDeputies

dfParty['averageperdeputy'] = partyAverageAmountPolitician = partyAverageAmount / partyDeputies

dfParty['averageperrefundperdeputy'] = partyAverageAmountRefundPolitician = (partyAverageAmount / partyDeputies) / partyNumberRefunds

dfParty['partyage'] = partyAge = df.groupby(['political_party'])['party_regdate'].unique()

dfParty['nmembers'] = partyMemberCount = df.groupby(['political_party'])['party_nmembers'].unique()

dfParty['position'] = partyPosition = df.groupby(['political_party'])['party_position'].unique()

dfParty['ideology1'] = partyIdeology1 = df.groupby(['political_party'])['party_ideology1'].unique()

dfParty['ideology2'] = partyIdeology2 = df.groupby(['political_party'])['party_ideology2'].unique()

dfParty['ideology3'] = partyIdeology3 = df.groupby(['political_party'])['party_ideology3'].unique()

dfParty['ideology4'] = partyIdeology4 = df.groupby(['political_party'])['party_ideology4'].unique()

print(dfParty.sort_values(by=['nrefundsdeputy'], ascending=False))



# In general, party data does not suggest any structural misconduct. 

# Parties with small numbers of deputies declare the most per capita;

# The larger the number of deputies gets, the more types and the lower the average per capita.

##

## Amount refunded

##

# Average amount per refund per politician is low except for extremely small parties

# PTN, PRB, SD and PTB all are above 40.000 per politician in total amount refunded

##

## Number of refunds

##

# No specific parties present themselves as investigation targets directly, although the PT and PMDB both have huge numbers of transactions

# Based on the number of refunds per deputy, PSL, PV, PHS and REDE (all small parties) seem to be gaming the system. PT, as first major party, comes in fifth.
##Visualizing which parties receive which amounts within each category of refunds

# It's to be expected there is spread within the amounts paid out for each category.

# Boxplot to visualize spending effects

#df['refund_description'] = df['refund_description'].apply(lambda x: x.decode("utf-8"))

partyRefundsVariance = sb.factorplot(kind='box', y='refund_value', x='refund_description', hue='political_party',

               data=df, size=20, aspect=1.5, legend_out=False) 

partyRefundsVariance.set_xticklabels(rotation=90)

partyRefundsVariance

# 

# Categories with no seemingly major deviations: 

# Fuel; 

# Ground vehicle rent; 

# Taxi / parking / toll; 

# Meal; 

# Hotel; 

# Security service; 

# Publication signature services; 

# Ground transportation ticket;

# Workshop / course / event;

# Ship / boat rent.

#

# Categories without a clear roof:

# Airplane rent

# Office maintenance

# Dissemination of parliament activity;

# Consulting, research and technical work costs

# Mail costs

# Phone costs

# Flight tickets

#

## Seems logical - virtually all costs in the first category have a reasonable maximum spend. There could still be high frequencies of declaration, though. 

# Branch off categories without a clear maximum spend for deeper analysis

df['refund_description'].unique()

dfNoCap = df[(df['refund_description']=='DISSEMINATION OF PARLIAMENTARY ACTIVITY') | (df['refund_description']=='CONSULTING, RESEARCH AND TECHNICAL WORK COSTS') | (df['refund_description']=='AIRPLANE RENT') | (df['refund_description']=='DISSEMINATION OF PARLIAMENTARY ACTIVITY') | (df['refund_description']=='CONSULTING, RESEARCH AND TECHNICAL WORK COSTS') | (df['refund_description']=='AIRPLANE RENT') | (df['refund_description']=='OFFICE MAINTENANCE') | (df['refund_description']=='MAIL COSTS') | (df['refund_description']=='PHONE COSTS') | (df['refund_description']=='FLIGHT TICKETS')]  

# Branch off categories with a clear maximum spend to examine high frequencies of similar payments

dfCap = df[(df['refund_description']=='FUEL COSTS') | (df['refund_description']=='GROUND VEHICLE RENT') | (df['refund_description']=='TAXI, PARKING AND TOLL COSTS') | (df['refund_description']=='MEAL COSTS') | (df['refund_description']=='HOTEL COSTS') | (df['refund_description']=='SECURITY SERVICE') | (df['refund_description']=='PUBLICATION SIGNATURE EXPENSES') | (df['refund_description']=='GROUND TRANSPORTATION TICKET') | (df['refund_description']=='WORKSHOP/COURSE/EVENT COSTS') | (df['refund_description']=='SHIP/BOAT RENT')]
## Use capped transactions to look for excessive numbers of declarations per deputy

# Get number of deputies per refund description

refundDeputy = dfCap.groupby(['refund_description'])['deputy_name'].nunique()

# Get number of refunds per refund description per deputy 

deputyRefund = dfCap.groupby(['deputy_name'])['refund_description'].value_counts()

deputyRefund.sort_values(ascending=False)

#Calculate timespan in days in which refunds could be given

refundDates = pd.Series(df['refund_date'].unique())

refundDates.sort_values(ascending=True)

refundDates = pd.to_datetime(refundDates.str[0:10], format = '%Y-%m-%d')

refundPeriod = refundDates.max() - refundDates.min()

refundPeriod.days

#View refunds with occurrence of over 439 times (multiple a day, on average)

deputyRefund = pd.DataFrame(deputyRefund)

deputyRefund = deputyRefund.rename(columns={'refund_description':'numberRefunds'})

excessiveRefunds = deputyRefund[deputyRefund['numberRefunds']>439]

#print deputyRefund[deputyRefund>376] #Assuming 6 days of work a week instead of the 7 above

# Look at amounts

excessiveRefundsNames = excessiveRefunds.index.get_level_values(0)

excessiveRefundsTypes = excessiveRefunds.index.get_level_values(1)

dfCapExc = dfCap[(dfCap['deputy_name'].isin(excessiveRefundsNames.values)) & (dfCap['refund_description'].isin(excessiveRefundsTypes.values))]

highRefundTotals= dfCapExc.groupby(['deputy_name','refund_description'])['refund_value'].sum()

highRefundTotals = highRefundTotals.astype(int)

highRefundTotals.plot.barh(figsize = (40,10))

# Examine statistics for companies paying, focus on fuel costs

excessiveRefundsCompanies = dfCapExc.groupby(['deputy_name','refund_description'])['company_name'].value_counts()

excessiveRefundsTotalValue = dfCapExc.groupby(['deputy_name','refund_description', 'company_name'])['refund_value'].sum()

excessiveRefundsMeanValue = dfCapExc.groupby(['deputy_name','refund_description', 'company_name'])['refund_value'].mean() 

print(excessiveRefundsCompanies.describe(), excessiveRefundsTotalValue.describe(), excessiveRefundsMeanValue.describe())
# Set some cutoffs to select cases which seem excessive

frequentRefunds = excessiveRefundsCompanies[excessiveRefundsCompanies > 40]

highValueRefunds = excessiveRefundsTotalValue[excessiveRefundsTotalValue > 10000]

highAvgValueRefunds = excessiveRefundsMeanValue[excessiveRefundsMeanValue > 1000]

# Examine refunds in category fuel for deputies with total refunds > 10000

deputyNames = highValueRefunds.index.get_level_values(0)

fuelRefunds = dfCap[(dfCap['deputy_name'].isin(deputyNames)) & (dfCap['refund_description']=='FUEL COSTS') & (dfCap['refund_value'] > 1000)]

fuelRefunds = pd.concat((fuelRefunds['deputy_name'], fuelRefunds['refund_date'], fuelRefunds['refund_value']), axis = 1)

# Liters of fuel bought for 4 reais a liter

fuelRefunds['liters_bought'] = fuelRefunds['refund_value']/4

# Number of Hummer H2 tanks that could be filled with that amount of gasoline

fuelRefunds['nHummer_tanks'] = fuelRefunds['liters_bought']/121

# Number of kilometers that could be driven in said hummer, assuming an infinite capacity gas tank, before gas would run out

fuelRefunds['km_driven_Hummer'] = fuelRefunds['nHummer_tanks']*877

# Number of hours of driving that would involve at 100 km/hour

fuelRefunds['hours_100_hour'] = fuelRefunds['km_driven_Hummer']/100

# Number of hours of driving per deputy based on total of large sum fuel refunds

totalHours = fuelRefunds.groupby(fuelRefunds['deputy_name'])['hours_100_hour'].sum()

# Number of full work days spent driving

longDrive = totalHours/8

# Proportion of declaration period spent driving

workdaysInDeclarationPeriod = round((439/7)*5) #Rough estimate

timeSpentDrivingPercent = longDrive / (workdaysInDeclarationPeriod / 100)

print (fuelRefunds, timeSpentDrivingPercent)

#Note that high amounts seem to recur monthly
timeSpentDrivingPercent.plot.bar(color = 'r')

plt.ylim(0,100)

plt.title('Percentage of workdays in declaration period spent driving per deputy, assuming\n a Hummer H2, total consumption of paid fuel and infinite capacity gas tank')

plt.xlabel('Deputy name')

plt.ylabel('Time spent driving in % of total time in declaration period')

plt.xticks((range(0,7,1)),rotation=90)

plt.margins(0.2)

plt.show()
### Examine hotel refunds

dfCapHotels = dfCap[(dfCap['refund_description']=='HOTEL COSTS')]

hotelCount = dfCapHotels.groupby(['deputy_name'])['refund_description'].count().sort_values(ascending=False)

hotelCostMax = dfCapHotels.groupby(['deputy_name'])['refund_value'].max().sort_values(ascending=False)

hotelCostAvg = dfCapHotels.groupby(['deputy_name'])['refund_value'].mean().sort_values(ascending=False)

hotelCostSum = dfCapHotels.groupby(['deputy_name'])['refund_value'].sum().sort_values(ascending=False)

manyHotels = hotelCount[hotelCount > 100]

expensiveHotels = hotelCount[(hotelCostAvg > 3000)|(hotelCostSum > 30000)]

print(manyHotels, expensiveHotels)

manyHotelsNames = manyHotels.index.get_level_values(0)

expensiveHotelNames = expensiveHotels.index.get_level_values(0)

dfExpensiveHotels = dfCap[(dfCap['deputy_name'].isin(expensiveHotelNames)) & (dfCap['refund_description']=='HOTEL COSTS') & ((dfCap['refund_value'].mean()>3000)|(dfCap['refund_value'].sum() > 30000))]

#for name in dfExpensiveHotels['deputy_name'].unique():

    #print(dfExpensiveHotels[dfExpensiveHotels['deputy_name']==name])