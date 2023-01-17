import numpy as np 

import pandas as pd 



import matplotlib.pyplot as plt

import seaborn as sns

from collections import defaultdict

import os
FILE_PATH = '/kaggle/input/startup-investments-crunchbase/investments_VC.csv'

df = pd.read_csv(FILE_PATH, encoding='ISO-8859-2')

#df.head(5)
# removing space from feature name

df = df.rename(columns={' market ': 'market', ' funding_total_usd ': 'funding_total_usd'})

# feature names

print(f"Features: {df.columns.values}\n")

print(f"Number of rows: {df.shape[0]}")

print(f"Number of cols: {df.shape[1]}")
# fill the missing value

df = df.dropna(subset=['name', 'country_code', 'status', 'market'])

df = df.fillna('None')
# function to convert string object to int64

def object_to_int(s): 

    return int(''.join(s.strip().split(',')))



# remove 

df.funding_total_usd = df.funding_total_usd.apply(lambda x: '0' if x == ' -   ' else x)

df.funding_total_usd = df.funding_total_usd.apply(lambda x: object_to_int(x))
# return the operating, acquired, closed percentage of startups in each market

def status_per_market(df, market, status):

    percentages = []

    for _, curr in enumerate(status):

        startup_status = df[(df['market'] == market) & (df['status'] == curr)].shape[0]

        total_startup = df[df['market'] == market].shape[0]

        percentages.append((startup_status/total_startup)*100)

    # list containing percent of operating, acquired, closed startups for each market

    return percentages

        

    

def funding_per_market(df, market):

    # tf_per_mkt : total funding for each market

    BILLION = 1000000000

    tf_all = df.funding_total_usd.sum()

    total_funding_per_mkt = df[(df['market'] == market)].funding_total_usd.sum()

    angel = df[(df['market'] == market)].angel.sum()

    vc = df[(df['market'] == market)].venture.sum()

    others = (total_funding_per_mkt - (angel + vc))/BILLION

    

    return total_funding_per_mkt/BILLION, (total_funding_per_mkt/tf_all)*100, angel/BILLION, vc/BILLION, others

    

    

def return_status_table(df, markets, status):

    print(f"Market/Startup\t\toperating\tacquired\tclosed")

    print(f"="*70)

    

    for market in markets: 

        spm = status_per_market(df, market, status)

        nmarket = market + ' ' * (max([len(m) for m in markets]) - len(market))

        

        print(f"{nmarket}\t\t{spm[0]:.2f}\t\t{spm[1]:.2f}\t\t{spm[2]:.2f}")

        

        

def return_funding_table(df, markets):

    print(f"Market/Startup\t\tfunding(B)\tfunding(%)\tangel(B)\tvc(B)")

    print(f"="*84)

    

    for market in markets: 

        tf, fpm, angel, vc, _ = funding_per_market(df, market)

        nmarket = market + ' ' * (max([len(m) for m in markets]) - len(market))

        

        print(f"{nmarket}\t\t{tf:.2f}\t\t{fpm:.2f}\t\t{angel:.2f}\t\t{vc:.2f}")
def pie_plot(titles, *args):

    # args: currently it takes in an input in the form of " a defaultdict wraped in a list" 

    explode = (0.0, 0.1, 0.1)

    _, ax = plt.subplots(1, len(args[0]), figsize=(20, 8))



    if len(args[0]) > 1:

        for i in range(len(args[0])):

            

            ax[i].set_title(titles[i])

            ax[i].pie(args[0][i]['sizes'], explode=explode, labels=args[0][i]['labels'], autopct='%1.1f%%',

                    shadow=True, startangle=90)

            ax[i].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        

        plt.show()





    elif len(args[0]) == 1:

        

        ax.set_title(titles)

        ax.pie(args[0]['sizes'], explode=explode, labels=args[0]['labels'], autopct='%1.1f%%',

            shadow=True, startangle=90)

        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        

        plt.show()        





def bar_plot(titles, *args):

    return





# plot pie charts for percentage of operating, acquired, closed

def status_plot(df, markets, d):

    #dic1 = {'labels': status, 'sizes': status_per_market(df, market, status)}

    pie_plot(markets, d)





def funding_plot(df, markets, d):

    pie_plot(markets, d)
def return_dict(df, markets, query):

    d = defaultdict(dict)   # a defaultdict of dict

    dd = []    # list to wrap around the above defaultdict





    for i, market in enumerate(markets):



        '''

        Step 1:     This function creates a defaultdict containing dictionary with labels and 

                    percentages in this case labels is the status (operating, closed etc.) of 

                    the company and the sizes contain the percentages corresponding to each status.



        Step 2:     Wrap the defaultdict created in the previous step with list.

        '''

        

        # step 1

        if query == 'status':

            d[i] = {'labels': ['operating', 'acquired', 'closed'], 'sizes': status_per_market(df, market, status)}

        

            # step 2

            dd.append(d[i])     # list of defualtdict



        if query == 'funding':

            _, _, angel, vc, other = funding_per_market(df, market)

            d[i] = {'labels': ['vc', 'angel', 'others'], 'sizes': [angel, vc, other]}

            dd.append(d[i])

    

    return dd
# Convert ' Games ' to 'Games' -- strip spaces

df.market = df.market.apply(lambda x: x.strip())



# top 20 markets with most startups

top40_market = list(df.market.value_counts()[:40].index)

status = ['operating', 'acquired', 'closed']

print(f"B = Billions\n")

return_status_table(df, top40_market, status)





markets = top40_market[:5]
print(f"B = Billions\n")

return_funding_table(df, top40_market)
markets = top40_market[:4]

status_dd = return_dict(df, markets, 'status')   

status_plot(df, markets, status_dd)
funding_dd = return_dict(df, markets, 'funding')

funding_plot(df, markets, funding_dd)
plt.figure(figsize=(16, 5))

df.founded_year.value_counts()[1:20].values

sns.barplot(y=df.founded_year.value_counts()[1:20].values, x=df.founded_year.value_counts()[1:20].index).set(title='Number of startups per year across all market')

plt.show()
# number of operating startups in a particular city

# percentage of operating startups in a particular city

# number of startups from a country

# funding rounds

# angel

# category list

# market that has the most failure, closed, operating

# total funding usd