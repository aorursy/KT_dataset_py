# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Cleveland demographic data



from datetime import datetime



year = 2002



data = pd.read_csv("/kaggle/input/cleveland/merged_estfile.csv")



data = data.loc[data['Year'] >= year]

data = data.loc[data['MNAME'] == 'Cleveland-Elyria-Mentor, OH Metro Area']

data = data.loc[data['Mode'] == 'Bus']



data = data[['Year', 'FARE_per_UPT_cleaned', 'Tot_Pop', 

             'Tot_Instate_Pop', 'Tot_Outstate_Pop', 'Tot_NonUSA_POP',

             'Total_Median_Income_Individual', 'Native_Instate_Med_Inc_Indiv', 

             'Native_Outstate_Med_Inc_Indiv', 'Native_Outcountry_Med_Inc_Indiv',

             'Total_Pop_Poverty', 'HH_MED_INC', 'PCT_HH_NO_VEH', 'JTW_HOME_PCT', 'TSD_POP_PCT_CBSA']]



data['Year'] = data['Year'].astype(str)



row_2019 = {'Year':'2019'}

data = data.append(row_2019, ignore_index=True)

row_2020 = {'Year':'2020'}

data = data.append(row_2020, ignore_index=True)



for i in range(1,len(data.columns)):

    prev1 = float(data.iloc[2018-year][i])

    prev2 = float(data.iloc[2017-year][i])

    diff = prev1-prev2

    col_val = prev1 + diff

    data.at[2019-year, data.columns[i]] = col_val

    data.at[2020-year, data.columns[i]] = 1

    

data['Year'] = pd.to_datetime(data['Year'])

data = data.drop([len(data) - 1])

data['Year'] = data['Year'].apply(lambda x: x.strftime('%Y'))

data = data.set_index('Year')

data = data.astype(float)





excel1 = pd.ExcelFile("/kaggle/input/cleveland/ClevelandAnnualGasPrice.xls")

gasprices = pd.read_excel(excel1, 'Data 1')

gasprices = gasprices.drop([0,1])

gasprices = gasprices.rename({'Back to Contents':'Year','Data 1: Cleveland, OH Regular All Formulations Retail Gasoline Prices (Dollars per Gallon)': 'GasPrice'}, axis=1)

gasprices['Year'] = gasprices['Year'].apply(lambda x: x.strftime('%Y'))

gasprices = gasprices.dropna()

gasprices = gasprices.set_index('Year')

combined = data.join(gasprices)



unemploy = pd.read_excel("/kaggle/input/cleveland/monthlyunemployment.xlsx")

unemploy = unemploy.loc[unemploy['Unnamed: 3'] == 'Cleveland-Elyria, OH MSA']

unemploy = unemploy.rename({

        "Unnamed: 4" : "year",

        "Unnamed: 5" : "month",

        "Unnamed: 6": "Civilian Labor Force",

        "Unnamed: 7" : "Employment",

        "Unnamed: 8" : "Unemployment",

        "Unnamed: 9" : "Unemployment Rate"}, axis=1)

unemploy['myDt'] = unemploy.apply(lambda row: datetime.strptime(f"{int(row.year)}-{int(row.month)}", '%Y-%m'), axis=1)

unemploy = unemploy.set_index('myDt')

unemploy = unemploy.drop(['Table 1. Civilian labor force and unemployment by metropolitan area, seasonally adjusted',

                          'Unnamed: 1',

                          'Unnamed: 2',

                          'Unnamed: 3',

                          'year',

                          'month'], axis=1)

unemploy = unemploy.astype(float)

unemploy = unemploy.resample('Y').mean()

unemploy = unemploy.reset_index()

unemploy['myDt'] = unemploy['myDt'].apply(lambda x: x.strftime('%Y'))



unemploy = unemploy.set_index('myDt')

combined = combined.join(unemploy)

combined = combined.dropna()





print(combined)

#quarterly 2009 - end

quarterly = combined.reset_index()

quarterly['Year'] = quarterly['Year'].astype(int)

quarterly = quarterly.loc[quarterly['Year'] >= 2009]

quarterly['Year'] = pd.to_datetime(quarterly['Year'], format='%Y')

quarterly = quarterly.set_index('Year')

quarterly = quarterly.resample('Q').mean()

idx = quarterly.index.to_period('Q')

quarterly.index = idx

quarterly = quarterly.reset_index()



row_2019Q2 = {'Year':'2019Q2'}

quarterly = quarterly.append(row_2019Q2, ignore_index=True)



row_2019Q3 = {'Year':'2019Q3'}

quarterly = quarterly.append(row_2019Q3, ignore_index=True)



row_2019Q4 = {'Year':'2019Q4'}

quarterly = quarterly.append(row_2019Q4, ignore_index=True)



start = 0

end = 4

track = True



while track:



    for i in range(1,len(quarterly.columns)):

        initial = float(quarterly.iloc[start, i])

        

        final = float(quarterly.iloc[end, i])

        roc = (final - initial)/4

        prev1 = initial

                

        for j in range(start+1,end):

            col_val = prev1 + roc

            col_name = quarterly.columns.values[i]

            quarterly.at[j, col_name]=col_val

            prev1 = col_val

            

    start = start+4

    end = end+4

    

    if end >= 43:

        break

        

start = 39

end = 40

for i in range(1, len(quarterly.columns)):

        initial = float(quarterly.iloc[start][i])

        final = float(quarterly.iloc[end][i])

        roc = (final - initial)

        prev1 = initial

        for j in range(41,44):

            col_val = prev1 + roc

            col_name = quarterly.columns.values[i]

            quarterly.at[j, col_name] = col_val

            prev1 = col_val

       



print(quarterly)
export_csv = quarterly.to_csv ('quarterlyclevelanddemodata.csv', index = False, header = True)

print("Export Complete!")
#monthly 2015 - end

monthly = combined.reset_index()

monthly['Year'] = monthly['Year'].astype(int)

monthly = monthly.loc[monthly['Year'] >= 2015]

monthly['Year'] = pd.to_datetime(monthly['Year'], format='%Y')

monthly = monthly.set_index('Year')

monthly = monthly.resample('M').mean()

monthly = monthly.reset_index()

monthly['Year'] = monthly['Year'].apply(lambda x: x.strftime('%Y-%m'))



for i in range(2,13):

    j = str(i)

    if i <10:

        name = '2019-0'+j

    else:

        name = '2019-'+j

    row = {'Year':name}

    monthly = monthly.append(row, ignore_index=True)

    

start = 0

end = 12

track = True



while track:



    for i in range(1,len(monthly.columns)):

        initial = float(monthly.iloc[start, i])

        

        final = float(monthly.iloc[end, i])

        roc = (final - initial)/12

        prev1 = initial

                

        for j in range(start+1,end):

            col_val = prev1 + roc

            col_name = monthly.columns.values[i]

            monthly.at[j, col_name]=col_val

            prev1 = col_val

            

    start = start+12

    end = end+12

    

    if end >= 49:

        break

        

start = 47

end = 48

for i in range(1, len(monthly.columns)):

        initial = float(monthly.iloc[start][i])

        final = float(monthly.iloc[end][i])

        roc = (final - initial)

        prev1 = initial

        for j in range(49,60):

            col_val = prev1 + roc

            col_name = monthly.columns.values[i]

            monthly.at[j, col_name] = col_val

            prev1 = col_val



print(monthly)
export_csv = monthly.to_csv ('monthlyclevelanddemodata.csv', index = False, header = True)

print("Export Complete!")