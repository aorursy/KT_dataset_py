#Global calls and variables

import requests
import json
import pandas as pd

all_countries = [{"country":"Afghanistan"},{"country":"Albania"},{"country":"Algeria"},{"country":"Angola"},
                 {"country":"Argentina"},{"country":"Armenia"},{"country":"Australia"},{"country":"Austria"},
                 {"country":"Azerbaijan"},{"country":"Bangladesh"},{"country":"Belarus"},{"country":"Belgium"},
                 {"country":"Benin"},{"country":"Bolivia"},{"country":"Bosnia and Herzegovina"},{"country":"Brazil"},
                 {"country":"Bulgaria"},{"country":"Burkina Faso"},{"country":"Cambodia"},{"country":"Cameroon"},
                 {"country":"Canada"},{"country":"Chile"},{"country":"Colombia"},{"country":"Costa Rica"},
                 {"country":"C\u00f4te d'Ivoire"},{"country":"Croatia"},{"country":"Czech Republic"},
                 {"country":"Denmark"},{"country":"Dominican Republic"},{"country":"Ecuador"},{"country":"Egypt"},
                 {"country":"El Salvador"},{"country":"Ethiopia"},{"country":"Finland"},{"country":"France"},
                 {"country":"Germany"},{"country":"Ghana"},{"country":"Greece"},{"country":"Guatemala"},
                 {"country":"Haiti"},{"country":"Honduras"},{"country":"Hong Kong"},{"country":"Hungary"},
                 {"country":"India"},{"country":"Indonesia"},{"country":"Iraq"},{"country":"Ireland"},
                 {"country":"Israel"},{"country":"Italy"},{"country":"Japan"},{"country":"Jordan"},
                 {"country":"Kazakhstan"},{"country":"Kenya"},{"country":"Kuwait"},{"country":"Kyrgyzstan"},
                 {"country":"Lebanon"},{"country":"Libya"},{"country":"Malaysia"},{"country":"Mali"},
                 {"country":"Mexico"},{"country":"Moldova"},{"country":"Morocco"},{"country":"Mozambique"},
                 {"country":"Myanmar"},{"country":"Nepal"},{"country":"Netherlands"},{"country":"New Zealand"},
                 {"country":"Nicaragua"},{"country":"Nigeria"},{"country":"Norway"},{"country":"Oman"},
                 {"country":"Pakistan"},{"country":"Palestine"},{"country":"Panama"},{"country":"Paraguay"},
                 {"country":"Peru"},{"country":"Philippines"},{"country":"Poland"},{"country":"Portugal"},
                 {"country":"Puerto Rico, U.S."},{"country":"Qatar"},{"country":"Romania"},{"country":"Russia"},
                 {"country":"Saudi Arabia"},{"country":"Senegal"},{"country":"Serbia"},{"country":"Singapore"},
                 {"country":"Slovakia"},{"country":"Slovenia"},{"country":"South Africa"},{"country":"South Korea"},
                 {"country":"Spain"},{"country":"Sri Lanka"},{"country":"Sudan"},{"country":"Sweden"},
                 {"country":"Switzerland"},{"country":"Taiwan"},{"country":"Tanzania"},{"country":"Thailand"},
                 {"country":"Tunisia"},{"country":"Turkey"},{"country":"Ukraine"},{"country":"United Arab Emirates"},
                 {"country":"United Kingdom"},{"country":"United States of America"},{"country":"Uruguay"},
                 {"country":"Uzbekistan"},{"country":"Venezuela"},{"country":"Vietnam"},{"country":"Yemen"}]

#Helper function to collect data
#There are no error checks included in this. Use it wisely

def get_data(indicator, d_type, wise, daterange='20200101-20201231'):
    """
    Collects the data from UMD API.

    Parameters (pass all as strings):
    indicator : Can be either 'flu' or 'covid'
    d_type    : Can be either 'daily' or 'smoothed'
    wise      : Can be either 'country' or 'country_region'
    daterange : 'yyyymmdd-yyyymmdd'
    
    Returns:
    df        : A dataframe of collected data
    (Also saves a CSV to current working directory)
    """

    df_list = []
    
    url_part1 = "https://covidmap.umd.edu/api/resources?indicator="+ indicator +"&type="+ d_type +"&country="
    
    if wise == 'country':
        url_part2 = "&daterange=" + daterange
    elif wise == 'country_region':
        url_part2 = "&region=all&daterange=" + daterange
        
    for country in all_countries:
        url =  url_part1 + country['country'] + url_part2
        res = requests.get(url)
        jsondata = json.loads(res.text)
        df_list.append(pd.DataFrame.from_records(jsondata['data']))

    df = pd.concat(df_list)
    csv_name = indicator + '_' + d_type + '_' + wise + '_wise.csv'
    df.to_csv(csv_name)
    return df
#Function calls to collect all data (One call for a file each)

get_data('covid','daily','country')
get_data('covid','daily','country_region')
get_data('covid','smoothed','country')
get_data('covid','smoothed','country_region')
get_data('flu','daily','country')
get_data('flu','daily','country_region')
get_data('flu','smoothed','country')
get_data('flu','smoothed','country_region')
#You can change this to gather data for a particular date range
#format = 'yyyymmdd-yyyymmdd'

daterange = '20200101-20201231'
# covid daily country-wise

df_list1 = []

for country in all_countries:
    res = requests.get("https://covidmap.umd.edu/api/\
resources?indicator=covid&type=daily&country=" + country['country'] + "&daterange=" + daterange)
    jsondata = json.loads(res.text)
    df_list.append(pd.DataFrame.from_records(jsondata['data']))
    
pd.concat(df_list1).to_csv('covid_daily_country_wise.csv')
# covid daily country-region-wise

df_list2 = []

for country in all_countries:
    res = requests.get("https://covidmap.umd.edu/api/\
resources?indicator=covid&type=daily&country=" + country['country'] + "&region=all&daterange=" + daterange)
    jsondata = json.loads(res.text)
    df_list.append(pd.DataFrame.from_records(jsondata['data']))
    
pd.concat(df_list2).to_csv('covid_daily_country_region_wise.csv')
# covid smoothed country-wise

df_list3 = []

for country in all_countries:
    res = requests.get("https://covidmap.umd.edu/api/\
resources?indicator=covid&type=smoothed&country=" + country['country'] + "&daterange=" + daterange)
    jsondata = json.loads(res.text)
    df_list2.append(pd.DataFrame.from_records(jsondata['data']))
    
pd.concat(df_list3).to_csv('covid_smoothed_country_wise.csv')
# covid smoothed country-region-wise

df_list4 = []

for country in all_countries:
    res = requests.get("https://covidmap.umd.edu/api/\
resources?indicator=covid&type=smoothed&country=" + country['country'] + "&region=all&daterange=" + daterange)
    jsondata = json.loads(res.text)
    df_list2.append(pd.DataFrame.from_records(jsondata['data']))
    
pd.concat(df_list4).to_csv('covid_smoothed_country_region_wise.csv')
# flu daily country-wise

df_list5 = []

for country in all_countries:
    res = requests.get("https://covidmap.umd.edu/api/\
resources?indicator=flu&type=daily&country=" + country['country'] + "&daterange=" + daterange)
    jsondata = json.loads(res.text)
    df_list3.append(pd.DataFrame.from_records(jsondata['data']))
    
pd.concat(df_list5).to_csv('flu_daily_country_wise.csv')
# flu daily country-region-wise

df_list6 = []

for country in all_countries:
    res = requests.get("https://covidmap.umd.edu/api/\
resources?indicator=flu&type=daily&country=" + country['country'] + "&region=all&daterange=" + daterange)
    jsondata = json.loads(res.text)
    df_list3.append(pd.DataFrame.from_records(jsondata['data']))
    
pd.concat(df_list6).to_csv('flu_daily_country_region_wise.csv')
#flu smoothed country-wise

df_list7 = []

for country in all_countries:
    res = requests.get("https://covidmap.umd.edu/api/\
resources?indicator=flu&type=smoothed&country=" + country['country'] + "&daterange=" + daterange)
    jsondata = json.loads(res.text)
    df_list4.append(pd.DataFrame.from_records(jsondata['data']))
    
pd.concat(df_list7).to_csv('flu_smoothed_country_wise.csv')
#flu smoothed country-region-wise

df_list8 = []

for country in all_countries:
    res = requests.get("https://covidmap.umd.edu/api/\
resources?indicator=flu&type=smoothed&country=" + country['country'] + "&region=all&daterange=" + daterange)
    jsondata = json.loads(res.text)
    df_list4.append(pd.DataFrame.from_records(jsondata['data']))
    
pd.concat(df_list8).to_csv('flu_smoothed_country_region_wise.csv')