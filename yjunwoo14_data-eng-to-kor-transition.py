# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import pandas as pd
!wget https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv

!wget https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Deaths.csv

!wget https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Recovered.csv
!ls
#read data and table cleaning process

conf_df = pd.read_csv('time_series_19-covid-Confirmed.csv')

deaths_df = pd.read_csv('time_series_19-covid-Deaths.csv')

recv_df = pd.read_csv('time_series_19-covid-Recovered.csv')



dates = ['1/22/20', '1/23/20', '1/24/20', '1/25/20', '1/26/20', '1/27/20', '1/28/20', 

         '1/29/20', '1/30/20', '1/31/20', '2/1/20', '2/2/20', '2/3/20', '2/4/20', 

         '2/5/20', '2/6/20', '2/7/20', '2/8/20', '2/9/20', '2/10/20', '2/11/20', '2/12/20', 

         '2/13/20', '2/14/20', '2/15/20', '2/16/20', '2/17/20', '2/18/20', '2/19/20',

         '2/20/20', '2/21/20', '2/22/20', '2/23/20', '2/24/20', '2/25/20', '2/26/20',

         '2/27/20', '2/28/20', '2/29/20', '3/1/20', '3/2/20', '3/3/20', '3/4/20']



conf_df_long = conf_df.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], 

                            value_vars=dates, var_name='Date', value_name='Confirmed')



deaths_df_long = deaths_df.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], 

                            value_vars=dates, var_name='Date', value_name='Deaths')



recv_df_long = recv_df.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], 

                            value_vars=dates, var_name='Date', value_name='Recovered')



full_table = pd.concat([conf_df_long, deaths_df_long['Deaths'], recv_df_long['Recovered']], 

                       axis=1, sort=False)

full_table.head()



# converting to proper data format

full_table['Date'] = pd.to_datetime(full_table['Date'])

full_table['Recovered'] = full_table['Recovered'].astype('int')



# replacing Mainland china with just China

full_table['Country/Region'] = full_table['Country/Region'].replace('Mainland China', 'China')



# filling missing values with 0 in columns ('Confirmed', 'Deaths', 'Recovered')

full_table[['Confirmed', 'Deaths', 'Recovered']] = full_table[['Confirmed', 'Deaths', 'Recovered']].fillna(0)

full_table[['Province/State']] = full_table[['Province/State']].fillna('NA')



# full table

full_table.head()
full_table['Country/Region'].unique() 
full_table["Country/Region"]= full_table["Country/Region"].replace({ 

    'China' : '??????', 'Thailand' : '??????', 'Japan' : '??????', 'South Korea' : '????????????', 'Taiwan' : '??????', 'US' : '??????'

  , 'Macau' : '?????????', 'Hong Kong' : '??????', 'Singapore' : '????????????', 'Vietnam' : '?????????', 'France' : '?????????'

  , 'Nepal' : '??????', 'Malaysia' : '???????????????', 'Canada' : '?????????', 'Australia' : '??????', 'Cambodia' : '????????????'

  , 'Sri Lanka' : '??????', 'Germany' : '??????', 'Finland' : '?????????', 'United Arab Emirates' : '??????????????????'

  , 'Philippines' : '?????????', 'India' : '??????', 'Italy' : '??????', 'UK' : '??????', 'Russia' : '?????????', 'Sweden' : '?????????'

  , 'Spain' : '?????????', 'Belgium' : '?????????', 'Egypt' : '?????????', 'Iran' : '??????', 'Others' : '??????????????????'

  , 'Israel' : '????????????', 'Lebanon' : '?????????', 'Iraq' : '?????????', 'Oman' : '??????', 'Afghanistan' : '??????????????????'

  , 'Bahrain' : '?????????', 'Kuwait' : '????????????',  'Algeria': '?????????',

       'Croatia' : '???????????????', 'Switzerland' : '?????????', 'Austria' : '???????????????', 'Pakistan':'????????????', 'Brazil':'?????????',

       'Georgia':'?????????', 'Greece':'?????????', 'North Macedonia' : '????????????????????????', 'Norway':'????????????', 'Romania':'????????????',

       'Denmark':'?????????', 'Estonia':'???????????????', 'Netherlands':'????????????', 'San Marino':'??????????????????', 'Belarus':'????????????',

       'Iceland':'???????????????', 'Lithuania':'???????????????', 'Mexico':'?????????', 'New Zealand':'????????????', 'Nigeria':'???????????????',

       'Ireland':'????????????', 'Luxembourg':'???????????????', 'Monaco':'?????????', 'Qatar':'?????????', 'Ecuador':'????????????',

       'Azerbaijan':'??????????????????', 'Czech Republic':'??????', 'Armenia':'???????????????', 'Dominican Republic':'?????????????????????',

       'Indonesia':'???????????????', 'Portugal':'????????????', 'Andorra':'?????????', 'Latvia':'????????????', 'Morocco':'?????????',

       'Saudi Arabia':'?????????????????????', 'Senegal':'?????????', 'Argentina':'???????????????', 'Chile':'??????', 'Jordan':'?????????',

       'Ukraine':'???????????????', 'Saint Barthelemy':'????????????????????????', 'Hungary':'?????????', 'Faroe Islands':'?????? ??????',

       'Gibraltar':'????????????', 'Liechtenstein':'????????????????????????', 'Poland':'?????????', 'Tunisia':'????????????'})

full_table['Country/Region'].unique() 
full_table['Province/State'].unique() 
full_table["Province/State"]= full_table["Province/State"].replace({

       'Anhui' : '????????????', 'Beijing' : '????????????', 'Chongqing' : '?????????', 'Fujian' : '?????????'

     , 'Gansu' : '?????????', 'Guangdong' : '?????????', 'Guangxi' : '?????? ?????? ?????????', 'Guizhou' : '???????????????'

     , 'Hainan' : '????????????', 'Hebei' : '????????????', 'Heilongjiang' : '???????????????', 'Henan' : '?????????'

     , 'Hubei' : '????????????', 'Hunan': '?????????', 'Inner Mongolia' : '????????? ?????????', 'Jiangsu' : '?????????'

     , 'Jiangxi' : '?????????', 'Jilin' : '?????????',  'Liaoning' : '????????????', 'Ningxia' : '?????? ????????? ?????????'

     , 'Qinghai' : '????????????', 'Shaanxi' : '?????????', 'Shandong' :'?????????','Shanghai' : '????????????'

     , 'Shanxi' : '?????????', 'Sichuan' : '?????????', 'Tianjin' : '?????????', 'Tibet' : '????????? ?????????', 'Xinjiang' : '?????? ????????? ?????????'

     , 'Yunnan' : '?????????', 'Zhejiang' : '?????????', 'Taiwan' : '??????', 'Seattle, WA' : '?????????, WA', 'Chicago, IL' : '?????????, IL'

     , 'Tempe, AZ' : '??????, ????????????', 'Macau' : '?????????', 'Hong Kong' : '??????', 'Toronto, ON' : '?????????, ON'

     , 'British Columbia' : '???????????????????????? ???', 'Orange, CA' : '?????????, CA', 'Los Angeles, CA' : '??????????????????, CA'

     , 'New South Wales' : '?????????????????????', 'Victoria' : '????????????', 'Queensland' : '???????????? ???', 'London, ON': '??????, ON'

     , 'Santa Clara, CA' : '???????????????, CA', 'South Australia' : '??????????????????????????????', 'Boston, MA' : '?????????, MA'

     , 'San Benito, CA' : '??? ?????????, CA', 'Madison, WI' : '?????????, WI', 'Diamond Princess cruise ship' : '??????????????? ???????????????'

     , 'San Diego County, CA' : '??????????????? ???', 'San Antonio, TX' : '???????????????, TX' 

     , 'Omaha, NE (From Diamond Princess)' : '?????????, NE (??????????????? ???????????????)', 'Sacramento County, CA' : '??????????????? ?????????, CA'

     , 'Travis, CA (From Diamond Princess)' : '???????????? ????????????, CA (??????????????? ???????????????)'

     , 'From Diamond Princess' : '??????????????? ???????????????', 'Humboldt County, CA' : '????????? ?????????, CA'

     , 'Lackland, TX (From Diamond Princess)' : '????????? ????????????, TX (??????????????? ???????????????)'

     , 'Unassigned Location (From Diamond Princess)' : '???????????? (??????????????? ???????????????)', 

       ' Montreal, QC':'????????????, QC', 'Western Australia':'???????????????????????????????????????',

       'Snohomish County, WA':'??????????????? ???,WA', 'Providence, RI':'?????????????????????, RI', 'Tasmania':'???????????????????????????',

       'Grafton County, NH':'???????????? ?????????, NH', 'Hillsborough, FL':'???????????? ?????????, FL', 'New York City, NY':'??????, NY',

       'Placer County, CA':'???????????? ?????????, CA', 'San Mateo, CA':'?????????????????????, CA', 'Sarasota, FL':'??????????????????, FL',

       'Sonoma County, CA':'????????? ?????????, CA', 'Umatilla, OR':'????????????, OR', 'Fulton County, GA':'?????? ?????????, GA',

       'Washington County, OR':'????????? ?????????, OR', ' Norfolk County, MA': '?????? ?????????, MA', 'Berkeley, CA':'?????????, CA',

       'Maricopa County, AZ':'?????????????????? ???, AZ', 'Wake County, NC':'????????? ?????????, NC', 'Westchester County, NY':'?????????????????? ???, NY',

       'Orange County, CA':'????????? ?????????, CA', 'Northern Territory':'?????? ??????',

       'Contra Costa County, CA':'???????????????????????? ?????????, CA'})

full_table['Province/State'].unique()
full_table.rename(columns={'Province/State' : '????????????', 'Country/Region' : '??????/??????', 'Lat' : '??????', 'Long' : '??????'

                   , 'Date' : '??????','Confirmed' :'?????????', 'Deaths': '?????????', 'Recovered' :'?????????'}, inplace=True)

full_table.head()
full_table[(full_table['????????????'] == '???????????? (??????????????? ???????????????)') |  (full_table['????????????'] == '???????????? (??????????????? ???????????????)') 

        |  (full_table['????????????'] == '????????? ????????????, TX (??????????????? ???????????????)') |  (full_table['????????????'] == '???????????? ????????????, CA (??????????????? ???????????????)') 

        |  (full_table['????????????'] == '?????????, NE (??????????????? ???????????????)')]
full_table.to_csv("COVID-19_Korean.csv",index = False)