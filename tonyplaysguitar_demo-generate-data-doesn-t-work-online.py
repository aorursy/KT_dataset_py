import pandas as pd

import numpy as np

import requests
SteamSpyData = requests.get('http://steamspy.com/api.php',params={'request':'all'}).json()

SteamSpyDF = pd.DataFrame(list(SteamSpyData.values()))
from bs4 import BeautifulSoup

import time, timeit, re, sys, datetime
def getUrlList(df=pd.DataFrame([],columns=['appid','ksid','ksurl']).astype({'appid': np.uint64, 'ksid': np.uint64}),url='https://www.kickstarter.com/play'):

    # Visit the site and parse its contents using BeautifulSoup

    KSlinks= BeautifulSoup(requests.get(url).content, "html.parser")

    # Find the pieces with the project links

    linkGames = KSlinks.find_all('div',"NS_site__spotlight_project")

    rows = []

    duplicates_header=1

    for ind in linkGames:

        # Each project block should have two links - one to the steamstore page, one to the KS project page

        links = [x['href'] for x in ind.find_all(href=True)]

        appid = int(re.search('(?<=app/)\d+',links[0]).group(0))

        if (appid in df.appid.values) or (appid in [row[0] for row in rows]):

            # Each KS page will have a "header" with the same highlighted projects, so we don't want to waste time scraping those over and over

            if duplicates_header:

                print('Duplicate IDs found: ',end='')

                duplicates_header=0

            print(appid,end=', ')

        else:

            try:

                # Take the link to the KS project page and parse it again with the BeautifulSoup

                KSpage = BeautifulSoup(requests.get('https://www.kickstarter.com'+links[-1]).content,"html.parser")

                # The project ID is appended to the classname "class=Project[ID]" of the data environment

                ksid = int(re.search('(?<=class="Project)\d+',str(KSpage.find('data'))).group(0))

                rows = rows+[[appid,ksid,links[-1]]]

                time.sleep(0.1) # just to avoid being punished by KS

            except:

                print('\nError loading data: ',links[-1],sys.exc_info()[0:2])

                

    return pd.concat([df,pd.DataFrame(rows,columns=['appid','ksid','ksurl']).astype({'appid': np.uint64, 'ksid': np.uint64})])
# It's a good idea to print out some information on the progress of the algorithm

start_time = timeit.default_timer()



games_found = 0

page_number=1

# Default call of the crawler function will look at page 1 and generate the new dataframe

KSTable = getUrlList()

print('\nDone with page 1 in',timeit.default_timer() - start_time,'seconds')

while KSTable.shape[0]>games_found:

    # Following calls need to provide the next page and reuse the dataframe so that it's concatenated

    page_number+=1

    games_found=KSTable.shape[0]

    KSTable = getUrlList(KSTable,'https://www.kickstarter.com/play?page='+str(page_number))

    print('\nDone with page',page_number,'in',timeit.default_timer() - start_time,'seconds')

    

elapsed = timeit.default_timer() - start_time

print('Crawler finished in',elapsed,'seconds.',KSTable.shape[0],'games found.') # It took ~15 minutes in January 2018 to crawl 223 games.
missingDBentries=[]



# Determine the missing appid values in the SteamSpy request and check the appdetails request for each

for missingappid in set(KSTable.appid.astype('str').values)-set(SteamSpyData.keys()):

    missingitem = requests.get('http://steamspy.com/api.php',params={'request':'appdetails','appid':missingappid}).json()

    # Make sure there actually was some data

    if missingitem.get('name',{'name':None})!=None:

        missingDBentries+=[missingitem]



# If we got new entries, we can update the database.

# The appdetails request contains CCU and tags fields, which are not present in the global request,

# so we need to make sure to not accidentally create extra columns       

if len(missingDBentries)>0:

    SteamSpyMissing = pd.DataFrame(missingDBentries,columns = SteamSpyDF.columns)

    SteamSpyDF=SteamSpyDF.append(SteamSpyMissing,ignore_index=True)
start_time = timeit.default_timer()



start_date = datetime.datetime.strptime('2009-11-01','%Y-%m-%d')

week_count = (datetime.datetime.today() - start_date).days//7

currency_history_list = []

sleep_timer = 1

for single_date in (start_date + datetime.timedelta(7*n) for n in range(week_count)):

    while True:

        try:

            exchange_date = requests.get('https://api.fixer.io/' + single_date.strftime('%Y-%m-%d'),params = {'base':'USD'}).json()

            exchange_line = exchange_date['rates']

            exchange_line['date'] = exchange_date['date']

            currency_history_list+=[exchange_line]

            time.sleep(0.3)

            sleep_timer=1

            print(exchange_date['date'],end=', ')

            break

        except:

            print('\nError loading data: ',exchange_date['date'],' waiting',sleep_timer,'seconds to retry')

            time.sleep(sleep_timer)

            sleep_timer*=2

            continue

    

elapsed = timeit.default_timer() - start_time

Currencies = pd.DataFrame(currency_history_list,columns = ['date', 'AUD', 'ZAR', 'MYR', 'SGD', 'EUR', 'KRW', 'HRK', 'TRY', 'RUB','HKD', 'INR', 'CZK', 'LVL', 'LTL', 'SEK', 'IDR', 'JPY', 'GBP', 'MXN','DKK', 'HUF', 'RON', 'BGN', 'NZD', 'CAD', 'PLN', 'NOK', 'THB', 'PHP','ILS', 'CHF', 'CNY', 'BRL'])

print('\nCrawler finished in',elapsed,'seconds.',Currencies.shape[0],'entries with currency exchange rates.')
SteamSpyDF.to_csv('SteamSpy.csv',index=False)

KSTable.to_csv('KSreleases.csv',columns=['appid','ksid','ksurl'],index=False)

Currencies.to_csv('Currencies.csv',index=False)
import sqlite3
conn = sqlite3.connect('KS-Steam-Connection-'+datetime.datetime.today().strftime('%Y%m')+'.sqlite')
SteamSpyDF.to_sql('SteamSpy',conn,if_exists='replace',index=False)

KSTable.to_sql('KSreleased',conn,if_exists='replace',index=False)

Currencies.to_sql('Currencies',conn,if_exists='replace',index=False)
conn.close()