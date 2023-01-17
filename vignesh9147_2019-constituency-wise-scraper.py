# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

results_df1 = pd.DataFrame()

results_df2 = pd.DataFrame()

results_df3 = pd.DataFrame()

results_df4 = pd.DataFrame()

results_df5 = pd.DataFrame()

results_df6 = pd.DataFrame()

results_df7 = pd.DataFrame()

results_df8 = pd.DataFrame()

results_df9 = pd.DataFrame()

results_df10 = pd.DataFrame()

results_df11 = pd.DataFrame()

results_df12 = pd.DataFrame()

results_df13 = pd.DataFrame()

results_df14 = pd.DataFrame()

results_df15 = pd.DataFrame()

results_df16 = pd.DataFrame()

results_df17 = pd.DataFrame()

results_df18 = pd.DataFrame()

results_df19 = pd.DataFrame()

results_df20 = pd.DataFrame()

results_df21 = pd.DataFrame()

results_df22 = pd.DataFrame()

results_df23 = pd.DataFrame()

results_df24 = pd.DataFrame()

results_df25 = pd.DataFrame()

results_df26 = pd.DataFrame()

results_df27 = pd.DataFrame()

results_df28 = pd.DataFrame()

results_df29 = pd.DataFrame()

results_df30 = pd.DataFrame()

results_df31 = pd.DataFrame()

results_df32 = pd.DataFrame()

results_df33 = pd.DataFrame()

results_df34 = pd.DataFrame()

results_df35 = pd.DataFrame()

results_df36 = pd.DataFrame()

results_df37 = pd.DataFrame()



#AndhraPradesh S01

url = 'https://results.eci.gov.in/pc/en/trends/statewiseS011.htm'



dfs = pd.read_html(url)

df = dfs[0]

print(df)

idx = df[df[0] == 'Constituency'].index[0]

cols = list(df.iloc[idx,:])



df.columns = cols

print(df.columns)



df = df[df['Const. No.'].notnull()]

df = df.loc[df['Const. No.'].str.isdigit()].reset_index(drop=True)

df = df.dropna(axis=1,how='all')



df['Leading Candidate'] = df['Leading Candidate'].str.split('i',expand=True)[0]

df['Leading Party'] = df['Leading Party'].str.split('iCurrent',expand=True)[0]

df['Trailing Party'] = df['Trailing Party'].str.split('iCurrent',expand=True)[0]

df['Trailing Candidate'] = df['Trailing Candidate'].str.split('iAssembly',expand=True)[0]



results_df1 = results_df1.append(df)

url_list = [2,3]

for x in url_list:

    url = 'https://results.eci.gov.in/pc/en/trends/statewiseS01%s.htm' %x

    print ('Processed %s' %url)

    dfs = pd.read_html(url)

    df = dfs[0]



    df.columns = cols



    df = df[df['Const. No.'].notnull()]

    df = df.loc[df['Const. No.'].str.isdigit()].reset_index(drop=True)

    df = df.dropna(axis=1,how='all')



    df['Leading Candidate'] = df['Leading Candidate'].str.split('i',expand=True)[0]

    df['Leading Party'] = df['Leading Party'].str.split('iCurrent',expand=True)[0]

    df['Trailing Party'] = df['Trailing Party'].str.split('iCurrent',expand=True)[0]

    df['Trailing Candidate'] = df['Trailing Candidate'].str.split('iAssembly',expand=True)[0]



    results_df1 = results_df1.append(df).reset_index(drop=True)

print(results_df1)

results_df1 = results_df1.loc[:, :'Status']

results_df1["state Code"]= 'S01'

results_df1["state name"]= 'Andhra Pradesh'





#ArunanchalPradesh S02

url = 'https://results.eci.gov.in/pc/en/trends/statewiseS021.htm'



dfs = pd.read_html(url)

df = dfs[0]

print(df)

idx = df[df[0] == 'Constituency'].index[0]

cols = list(df.iloc[idx,:])



df.columns = cols

print(df.columns)



df = df[df['Const. No.'].notnull()]

df = df.loc[df['Const. No.'].str.isdigit()].reset_index(drop=True)

df = df.dropna(axis=1,how='all')



df['Leading Candidate'] = df['Leading Candidate'].str.split('i',expand=True)[0]

df['Leading Party'] = df['Leading Party'].str.split('iCurrent',expand=True)[0]

df['Trailing Party'] = df['Trailing Party'].str.split('iCurrent',expand=True)[0]

df['Trailing Candidate'] = df['Trailing Candidate'].str.split('iAssembly',expand=True)[0]



results_df2 = results_df2.append(df)

results_df2 = results_df2.loc[:, :'Status']

results_df2["state Code"]= 'S02'

results_df2["state name"]= 'Arunanchal Pradesh'





#Assam S03

url_list = [1,2]

for x in url_list:

    url = 'https://results.eci.gov.in/pc/en/trends/statewiseS03%s.htm' %x

    print ('Processed %s' %url)

    dfs = pd.read_html(url)

    df = dfs[0]



    df.columns = cols



    df = df[df['Const. No.'].notnull()]

    df = df.loc[df['Const. No.'].str.isdigit()].reset_index(drop=True)

    df = df.dropna(axis=1,how='all')



    df['Leading Candidate'] = df['Leading Candidate'].str.split('i',expand=True)[0]

    df['Leading Party'] = df['Leading Party'].str.split('iCurrent',expand=True)[0]

    df['Trailing Party'] = df['Trailing Party'].str.split('iCurrent',expand=True)[0]

    df['Trailing Candidate'] = df['Trailing Candidate'].str.split('iAssembly',expand=True)[0]



    results_df3 = results_df3.append(df).reset_index(drop=True)

results_df3 = results_df3.loc[:, :'Status']

results_df3["state Code"]= 'S03'

results_df3["state name"]= 'Assam'

print(results_df3)



#Bihar S04

url_list = [1,2,3,4]

for x in url_list:

    url = 'https://results.eci.gov.in/pc/en/trends/statewiseS04%s.htm' %x

    print ('Processed %s' %url)

    dfs = pd.read_html(url)

    df = dfs[0]



    df.columns = cols



    df = df[df['Const. No.'].notnull()]

    df = df.loc[df['Const. No.'].str.isdigit()].reset_index(drop=True)

    df = df.dropna(axis=1,how='all')



    df['Leading Candidate'] = df['Leading Candidate'].str.split('i',expand=True)[0]

    df['Leading Party'] = df['Leading Party'].str.split('iCurrent',expand=True)[0]

    df['Trailing Party'] = df['Trailing Party'].str.split('iCurrent',expand=True)[0]

    df['Trailing Candidate'] = df['Trailing Candidate'].str.split('iAssembly',expand=True)[0]



    results_df4 = results_df4.append(df).reset_index(drop=True)

results_df4 = results_df4.loc[:, :'Status']

results_df4["state Code"]= 'S04'

results_df4["state name"]= 'Bihar'



#Goa S05

url_list = [1]

for x in url_list:

    url = 'https://results.eci.gov.in/pc/en/trends/statewiseS05%s.htm' %x

    print ('Processed %s' %url)

    dfs = pd.read_html(url)

    df = dfs[0]



    df.columns = cols



    df = df[df['Const. No.'].notnull()]

    df = df.loc[df['Const. No.'].str.isdigit()].reset_index(drop=True)

    df = df.dropna(axis=1,how='all')



    df['Leading Candidate'] = df['Leading Candidate'].str.split('i',expand=True)[0]

    df['Leading Party'] = df['Leading Party'].str.split('iCurrent',expand=True)[0]

    df['Trailing Party'] = df['Trailing Party'].str.split('iCurrent',expand=True)[0]

    df['Trailing Candidate'] = df['Trailing Candidate'].str.split('iAssembly',expand=True)[0]



    results_df5 = results_df5.append(df).reset_index(drop=True)

results_df5 = results_df5.loc[:, :'Status']

results_df5["state Code"]= 'S05'

results_df5["state name"]= 'Goa'





#Gujarat S06

url_list = [1,2,3]

for x in url_list:

    url = 'https://results.eci.gov.in/pc/en/trends/statewiseS06%s.htm' %x

    print ('Processed %s' %url)

    dfs = pd.read_html(url)

    df = dfs[0]



    df.columns = cols



    df = df[df['Const. No.'].notnull()]

    df = df.loc[df['Const. No.'].str.isdigit()].reset_index(drop=True)

    df = df.dropna(axis=1,how='all')



    df['Leading Candidate'] = df['Leading Candidate'].str.split('i',expand=True)[0]

    df['Leading Party'] = df['Leading Party'].str.split('iCurrent',expand=True)[0]

    df['Trailing Party'] = df['Trailing Party'].str.split('iCurrent',expand=True)[0]

    df['Trailing Candidate'] = df['Trailing Candidate'].str.split('iAssembly',expand=True)[0]



    results_df6 = results_df6.append(df).reset_index(drop=True)

results_df6 = results_df6.loc[:, :'Status']

results_df6["state Code"]= 'S06'

results_df6["state name"]= 'Gujarat'



#Haryana S07

url_list = [1]

for x in url_list:

    url = 'https://results.eci.gov.in/pc/en/trends/statewiseS07%s.htm' %x

    print ('Processed %s' %url)

    dfs = pd.read_html(url)

    df = dfs[0]



    df.columns = cols



    df = df[df['Const. No.'].notnull()]

    df = df.loc[df['Const. No.'].str.isdigit()].reset_index(drop=True)

    df = df.dropna(axis=1,how='all')



    df['Leading Candidate'] = df['Leading Candidate'].str.split('i',expand=True)[0]

    df['Leading Party'] = df['Leading Party'].str.split('iCurrent',expand=True)[0]

    df['Trailing Party'] = df['Trailing Party'].str.split('iCurrent',expand=True)[0]

    df['Trailing Candidate'] = df['Trailing Candidate'].str.split('iAssembly',expand=True)[0]



    results_df7 = results_df7.append(df).reset_index(drop=True)

results_df7 = results_df7.loc[:, :'Status']

results_df7["state Code"]= 'S07'

results_df7["state name"]= 'Haryana'



#Himachal Pradesh

url_list = [1]

for x in url_list:

    url = 'https://results.eci.gov.in/pc/en/trends/statewiseS08%s.htm' %x

    print ('Processed %s' %url)

    dfs = pd.read_html(url)

    df = dfs[0]



    df.columns = cols



    df = df[df['Const. No.'].notnull()]

    df = df.loc[df['Const. No.'].str.isdigit()].reset_index(drop=True)

    df = df.dropna(axis=1,how='all')



    df['Leading Candidate'] = df['Leading Candidate'].str.split('i',expand=True)[0]

    df['Leading Party'] = df['Leading Party'].str.split('iCurrent',expand=True)[0]

    df['Trailing Party'] = df['Trailing Party'].str.split('iCurrent',expand=True)[0]

    df['Trailing Candidate'] = df['Trailing Candidate'].str.split('iAssembly',expand=True)[0]



    results_df8 = results_df8.append(df).reset_index(drop=True)

results_df8 = results_df8.loc[:, :'Status']

results_df8["state Code"]= 'S08'

results_df8["state name"]= 'Himachal Pradesh'



#Jammu & Kashmir S09

url_list = [1]

for x in url_list:

    url = 'https://results.eci.gov.in/pc/en/trends/statewiseS09%s.htm' %x

    print ('Processed %s' %url)

    dfs = pd.read_html(url)

    df = dfs[0]



    df.columns = cols



    df = df[df['Const. No.'].notnull()]

    df = df.loc[df['Const. No.'].str.isdigit()].reset_index(drop=True)

    df = df.dropna(axis=1,how='all')



    df['Leading Candidate'] = df['Leading Candidate'].str.split('i',expand=True)[0]

    df['Leading Party'] = df['Leading Party'].str.split('iCurrent',expand=True)[0]

    df['Trailing Party'] = df['Trailing Party'].str.split('iCurrent',expand=True)[0]

    df['Trailing Candidate'] = df['Trailing Candidate'].str.split('iAssembly',expand=True)[0]



    results_df9 = results_df9.append(df).reset_index(drop=True)

results_df9 = results_df9.loc[:, :'Status']

results_df9["state Code"]= 'S09'

results_df9["state name"]= 'Jammu & Kashmir'



#Karnataka S10

url_list = [1,2,3]

for x in url_list:

    url = 'https://results.eci.gov.in/pc/en/trends/statewiseS10%s.htm' %x

    print ('Processed %s' %url)

    dfs = pd.read_html(url)

    df = dfs[0]



    df.columns = cols



    df = df[df['Const. No.'].notnull()]

    df = df.loc[df['Const. No.'].str.isdigit()].reset_index(drop=True)

    df = df.dropna(axis=1,how='all')



    df['Leading Candidate'] = df['Leading Candidate'].str.split('i',expand=True)[0]

    df['Leading Party'] = df['Leading Party'].str.split('iCurrent',expand=True)[0]

    df['Trailing Party'] = df['Trailing Party'].str.split('iCurrent',expand=True)[0]

    df['Trailing Candidate'] = df['Trailing Candidate'].str.split('iAssembly',expand=True)[0]



    results_df10 = results_df10.append(df).reset_index(drop=True)

results_df10 = results_df10.loc[:, :'Status']

results_df10["state Code"]= 'S10'

results_df10["state name"]= 'Karnataka'



#Kerala S11

url_list = [1,2]

for x in url_list:

    url = 'https://results.eci.gov.in/pc/en/trends/statewiseS11%s.htm' %x

    print ('Processed %s' %url)

    dfs = pd.read_html(url)

    df = dfs[0]



    df.columns = cols



    df = df[df['Const. No.'].notnull()]

    df = df.loc[df['Const. No.'].str.isdigit()].reset_index(drop=True)

    df = df.dropna(axis=1,how='all')



    df['Leading Candidate'] = df['Leading Candidate'].str.split('i',expand=True)[0]

    df['Leading Party'] = df['Leading Party'].str.split('iCurrent',expand=True)[0]

    df['Trailing Party'] = df['Trailing Party'].str.split('iCurrent',expand=True)[0]

    df['Trailing Candidate'] = df['Trailing Candidate'].str.split('iAssembly',expand=True)[0]



    results_df11 = results_df11.append(df).reset_index(drop=True)

results_df11 = results_df11.loc[:, :'Status']

results_df11["state Code"]= 'S11'

results_df11["state name"]= 'Kerala'



#Madhya Pradesh S12

url_list = [1,2,3]

for x in url_list:

    url = 'https://results.eci.gov.in/pc/en/trends/statewiseS12%s.htm' %x

    print ('Processed %s' %url)

    dfs = pd.read_html(url)

    df = dfs[0]



    df.columns = cols



    df = df[df['Const. No.'].notnull()]

    df = df.loc[df['Const. No.'].str.isdigit()].reset_index(drop=True)

    df = df.dropna(axis=1,how='all')



    df['Leading Candidate'] = df['Leading Candidate'].str.split('i',expand=True)[0]

    df['Leading Party'] = df['Leading Party'].str.split('iCurrent',expand=True)[0]

    df['Trailing Party'] = df['Trailing Party'].str.split('iCurrent',expand=True)[0]

    df['Trailing Candidate'] = df['Trailing Candidate'].str.split('iAssembly',expand=True)[0]



    results_df12 = results_df12.append(df).reset_index(drop=True)

results_df12 = results_df12.loc[:, :'Status']

results_df12["state Code"]= 'S12'

results_df12["state name"]= 'Madhya Pradesh'



#Maharashtra S13

url_list = [1,2,3,4,5]

for x in url_list:

    url = 'https://results.eci.gov.in/pc/en/trends/statewiseS13%s.htm' %x

    print ('Processed %s' %url)

    dfs = pd.read_html(url)

    df = dfs[0]



    df.columns = cols



    df = df[df['Const. No.'].notnull()]

    df = df.loc[df['Const. No.'].str.isdigit()].reset_index(drop=True)

    df = df.dropna(axis=1,how='all')



    df['Leading Candidate'] = df['Leading Candidate'].str.split('i',expand=True)[0]

    df['Leading Party'] = df['Leading Party'].str.split('iCurrent',expand=True)[0]

    df['Trailing Party'] = df['Trailing Party'].str.split('iCurrent',expand=True)[0]

    df['Trailing Candidate'] = df['Trailing Candidate'].str.split('iAssembly',expand=True)[0]



    results_df13 = results_df13.append(df).reset_index(drop=True)

results_df13 = results_df13.loc[:, :'Status']

results_df13["state Code"]= 'S13'

results_df13["state name"]= 'Maharashtra'



#Manipur S14

url_list = [1]

for x in url_list:

    url = 'https://results.eci.gov.in/pc/en/trends/statewiseS14%s.htm' %x

    print ('Processed %s' %url)

    dfs = pd.read_html(url)

    df = dfs[0]



    df.columns = cols



    df = df[df['Const. No.'].notnull()]

    df = df.loc[df['Const. No.'].str.isdigit()].reset_index(drop=True)

    df = df.dropna(axis=1,how='all')



    df['Leading Candidate'] = df['Leading Candidate'].str.split('i',expand=True)[0]

    df['Leading Party'] = df['Leading Party'].str.split('iCurrent',expand=True)[0]

    df['Trailing Party'] = df['Trailing Party'].str.split('iCurrent',expand=True)[0]

    df['Trailing Candidate'] = df['Trailing Candidate'].str.split('iAssembly',expand=True)[0]



    results_df14 = results_df14.append(df).reset_index(drop=True)

results_df14 = results_df14.loc[:, :'Status']

results_df14["state Code"]= 'S14'

results_df14["state name"]= 'Manipur'



#Meghalaya S15

url_list = [1]

for x in url_list:

    url = 'https://results.eci.gov.in/pc/en/trends/statewiseS15%s.htm' %x

    print ('Processed %s' %url)

    dfs = pd.read_html(url)

    df = dfs[0]



    df.columns = cols



    df = df[df['Const. No.'].notnull()]

    df = df.loc[df['Const. No.'].str.isdigit()].reset_index(drop=True)

    df = df.dropna(axis=1,how='all')



    df['Leading Candidate'] = df['Leading Candidate'].str.split('i',expand=True)[0]

    df['Leading Party'] = df['Leading Party'].str.split('iCurrent',expand=True)[0]

    df['Trailing Party'] = df['Trailing Party'].str.split('iCurrent',expand=True)[0]

    df['Trailing Candidate'] = df['Trailing Candidate'].str.split('iAssembly',expand=True)[0]



    results_df15 = results_df15.append(df).reset_index(drop=True)

results_df15 = results_df15.loc[:, :'Status']

results_df15["state Code"]= 'S15'

results_df15["state name"]= 'Meghalaya'



#Mizoram S16

url_list = [1]

for x in url_list:

    url = 'https://results.eci.gov.in/pc/en/trends/statewiseS16%s.htm' %x

    print ('Processed %s' %url)

    dfs = pd.read_html(url)

    df = dfs[0]



    df.columns = cols



    df = df[df['Const. No.'].notnull()]

    df = df.loc[df['Const. No.'].str.isdigit()].reset_index(drop=True)

    df = df.dropna(axis=1,how='all')



    df['Leading Candidate'] = df['Leading Candidate'].str.split('i',expand=True)[0]

    df['Leading Party'] = df['Leading Party'].str.split('iCurrent',expand=True)[0]

    df['Trailing Party'] = df['Trailing Party'].str.split('iCurrent',expand=True)[0]

    df['Trailing Candidate'] = df['Trailing Candidate'].str.split('iAssembly',expand=True)[0]



    results_df16 = results_df16.append(df).reset_index(drop=True)

results_df16 = results_df16.loc[:, :'Status']

results_df16["state Code"]= 'S16'

results_df16["state name"]= 'Mizoram'



#Nagaland S17

url_list = [1]

for x in url_list:

    url = 'https://results.eci.gov.in/pc/en/trends/statewiseS17%s.htm' %x

    print ('Processed %s' %url)

    dfs = pd.read_html(url)

    df = dfs[0]



    df.columns = cols



    df = df[df['Const. No.'].notnull()]

    df = df.loc[df['Const. No.'].str.isdigit()].reset_index(drop=True)

    df = df.dropna(axis=1,how='all')



    df['Leading Candidate'] = df['Leading Candidate'].str.split('i',expand=True)[0]

    df['Leading Party'] = df['Leading Party'].str.split('iCurrent',expand=True)[0]

    df['Trailing Party'] = df['Trailing Party'].str.split('iCurrent',expand=True)[0]

    df['Trailing Candidate'] = df['Trailing Candidate'].str.split('iAssembly',expand=True)[0]



    results_df17 = results_df17.append(df).reset_index(drop=True)

results_df17 = results_df17.loc[:, :'Status']

results_df17["state Code"]= 'S17'

results_df17["state name"]= 'Nagaland'



#Odisha S18

url_list = [1,2,3]

for x in url_list:

    url = 'https://results.eci.gov.in/pc/en/trends/statewiseS18%s.htm' %x

    print ('Processed %s' %url)

    dfs = pd.read_html(url)

    df = dfs[0]



    df.columns = cols



    df = df[df['Const. No.'].notnull()]

    df = df.loc[df['Const. No.'].str.isdigit()].reset_index(drop=True)

    df = df.dropna(axis=1,how='all')



    df['Leading Candidate'] = df['Leading Candidate'].str.split('i',expand=True)[0]

    df['Leading Party'] = df['Leading Party'].str.split('iCurrent',expand=True)[0]

    df['Trailing Party'] = df['Trailing Party'].str.split('iCurrent',expand=True)[0]

    df['Trailing Candidate'] = df['Trailing Candidate'].str.split('iAssembly',expand=True)[0]



    results_df18 = results_df18.append(df).reset_index(drop=True)

results_df18 = results_df18.loc[:, :'Status']

results_df18["state Code"]= 'S18'

results_df18["state name"]= 'Odisha'



#Punjab S19

url_list = [1,2]

for x in url_list:

    url = 'https://results.eci.gov.in/pc/en/trends/statewiseS19%s.htm' %x

    print ('Processed %s' %url)

    dfs = pd.read_html(url)

    df = dfs[0]



    df.columns = cols



    df = df[df['Const. No.'].notnull()]

    df = df.loc[df['Const. No.'].str.isdigit()].reset_index(drop=True)

    df = df.dropna(axis=1,how='all')



    df['Leading Candidate'] = df['Leading Candidate'].str.split('i',expand=True)[0]

    df['Leading Party'] = df['Leading Party'].str.split('iCurrent',expand=True)[0]

    df['Trailing Party'] = df['Trailing Party'].str.split('iCurrent',expand=True)[0]

    df['Trailing Candidate'] = df['Trailing Candidate'].str.split('iAssembly',expand=True)[0]



    results_df19 = results_df19.append(df).reset_index(drop=True)

results_df19 = results_df19.loc[:, :'Status']

results_df19["state Code"]= 'S19'

results_df19["state name"]= 'Punjab'



#Rajasthan S20

url_list = [1,2,3]

for x in url_list:

    url = 'https://results.eci.gov.in/pc/en/trends/statewiseS20%s.htm' %x

    print ('Processed %s' %url)

    dfs = pd.read_html(url)

    df = dfs[0]



    df.columns = cols



    df = df[df['Const. No.'].notnull()]

    df = df.loc[df['Const. No.'].str.isdigit()].reset_index(drop=True)

    df = df.dropna(axis=1,how='all')



    df['Leading Candidate'] = df['Leading Candidate'].str.split('i',expand=True)[0]

    df['Leading Party'] = df['Leading Party'].str.split('iCurrent',expand=True)[0]

    df['Trailing Party'] = df['Trailing Party'].str.split('iCurrent',expand=True)[0]

    df['Trailing Candidate'] = df['Trailing Candidate'].str.split('iAssembly',expand=True)[0]



    results_df20 = results_df20.append(df).reset_index(drop=True)

results_df20 = results_df20.loc[:, :'Status']

results_df20["state Code"]= 'S20'

results_df20["state name"]= 'Rajasthan'



#Sikkim S21

url_list=[1]

for x in url_list:

    url = 'https://results.eci.gov.in/pc/en/trends/statewiseS21%s.htm' %x

    print ('Processed %s' %url)

    dfs = pd.read_html(url)

    df = dfs[0]



    df.columns = cols



    df = df[df['Const. No.'].notnull()]

    df = df.loc[df['Const. No.'].str.isdigit()].reset_index(drop=True)

    df = df.dropna(axis=1,how='all')



    df['Leading Candidate'] = df['Leading Candidate'].str.split('i',expand=True)[0]

    df['Leading Party'] = df['Leading Party'].str.split('iCurrent',expand=True)[0]

    df['Trailing Party'] = df['Trailing Party'].str.split('iCurrent',expand=True)[0]

    df['Trailing Candidate'] = df['Trailing Candidate'].str.split('iAssembly',expand=True)[0]



    results_df21 = results_df21.append(df).reset_index(drop=True)

results_df21 = results_df21.loc[:, :'Status']

results_df21["state Code"]= 'S21'

results_df21["state name"]= 'Sikkim'



#Tamilnadu S22

url_list = [1,2,3,4]

for x in url_list:

    url = 'https://results.eci.gov.in/pc/en/trends/statewiseS22%s.htm' %x

    print ('Processed %s' %url)

    dfs = pd.read_html(url)

    df = dfs[0]



    df.columns = cols



    df = df[df['Const. No.'].notnull()]

    df = df.loc[df['Const. No.'].str.isdigit()].reset_index(drop=True)

    df = df.dropna(axis=1,how='all')



    df['Leading Candidate'] = df['Leading Candidate'].str.split('i',expand=True)[0]

    df['Leading Party'] = df['Leading Party'].str.split('iCurrent',expand=True)[0]

    df['Trailing Party'] = df['Trailing Party'].str.split('iCurrent',expand=True)[0]

    df['Trailing Candidate'] = df['Trailing Candidate'].str.split('iAssembly',expand=True)[0]



    results_df22 = results_df22.append(df).reset_index(drop=True)

results_df22 = results_df22.loc[:, :'Status']

results_df22["state Code"]= 'S22'

results_df22["state name"]= 'Tamil Nadu'



#Tripura S23

url_list = [1]

for x in url_list:

    url = 'https://results.eci.gov.in/pc/en/trends/statewiseS23%s.htm' %x

    print ('Processed %s' %url)

    dfs = pd.read_html(url)

    df = dfs[0]



    df.columns = cols



    df = df[df['Const. No.'].notnull()]

    df = df.loc[df['Const. No.'].str.isdigit()].reset_index(drop=True)

    df = df.dropna(axis=1,how='all')



    df['Leading Candidate'] = df['Leading Candidate'].str.split('i',expand=True)[0]

    df['Leading Party'] = df['Leading Party'].str.split('iCurrent',expand=True)[0]

    df['Trailing Party'] = df['Trailing Party'].str.split('iCurrent',expand=True)[0]

    df['Trailing Candidate'] = df['Trailing Candidate'].str.split('iAssembly',expand=True)[0]



    results_df23 = results_df23.append(df).reset_index(drop=True)

results_df23 = results_df23.loc[:, :'Status']

results_df23["state Code"]= 'S23'

results_df23["state name"]= 'Tripura'



#UttarPradesh S24

url_list = [1,2,3,4,5,6,7,8]

for x in url_list:

    url = 'https://results.eci.gov.in/pc/en/trends/statewiseS24%s.htm' %x

    print ('Processed %s' %url)

    dfs = pd.read_html(url)

    df = dfs[0]



    df.columns = cols



    df = df[df['Const. No.'].notnull()]

    df = df.loc[df['Const. No.'].str.isdigit()].reset_index(drop=True)

    df = df.dropna(axis=1,how='all')



    df['Leading Candidate'] = df['Leading Candidate'].str.split('i',expand=True)[0]

    df['Leading Party'] = df['Leading Party'].str.split('iCurrent',expand=True)[0]

    df['Trailing Party'] = df['Trailing Party'].str.split('iCurrent',expand=True)[0]

    df['Trailing Candidate'] = df['Trailing Candidate'].str.split('iAssembly',expand=True)[0]



    results_df24 = results_df24.append(df).reset_index(drop=True)

results_df24 = results_df24.loc[:, :'Status']

results_df24["state Code"]= 'S24'

results_df24["state name"]= 'Uttar Pradesh'



#WestBengal S25

url_list = [1,2,3,4,5]

for x in url_list:

    url = 'https://results.eci.gov.in/pc/en/trends/statewiseS25%s.htm' %x

    print ('Processed %s' %url)

    dfs = pd.read_html(url)

    df = dfs[0]



    df.columns = cols



    df = df[df['Const. No.'].notnull()]

    df = df.loc[df['Const. No.'].str.isdigit()].reset_index(drop=True)

    df = df.dropna(axis=1,how='all')



    df['Leading Candidate'] = df['Leading Candidate'].str.split('i',expand=True)[0]

    df['Leading Party'] = df['Leading Party'].str.split('iCurrent',expand=True)[0]

    df['Trailing Party'] = df['Trailing Party'].str.split('iCurrent',expand=True)[0]

    df['Trailing Candidate'] = df['Trailing Candidate'].str.split('iAssembly',expand=True)[0]



    results_df25 = results_df25.append(df).reset_index(drop=True)

results_df25 = results_df25.loc[:, :'Status']

results_df25["state Code"]= 'S25'

results_df25["state name"]= 'West Bengal'



#Chhattisgarh S26

url_list = [1,2]

for x in url_list:

    url = 'https://results.eci.gov.in/pc/en/trends/statewiseS26%s.htm' %x

    print ('Processed %s' %url)

    dfs = pd.read_html(url)

    df = dfs[0]



    df.columns = cols



    df = df[df['Const. No.'].notnull()]

    df = df.loc[df['Const. No.'].str.isdigit()].reset_index(drop=True)

    df = df.dropna(axis=1,how='all')



    df['Leading Candidate'] = df['Leading Candidate'].str.split('i',expand=True)[0]

    df['Leading Party'] = df['Leading Party'].str.split('iCurrent',expand=True)[0]

    df['Trailing Party'] = df['Trailing Party'].str.split('iCurrent',expand=True)[0]

    df['Trailing Candidate'] = df['Trailing Candidate'].str.split('iAssembly',expand=True)[0]



    results_df26 = results_df26.append(df).reset_index(drop=True)

results_df26 = results_df26.loc[:, :'Status']

results_df26["state Code"]= 'S26'

results_df26["state name"]= 'Chhattisgarh'



#Jharkhand S27

url_list = [1,2]

for x in url_list:

    url = 'https://results.eci.gov.in/pc/en/trends/statewiseS27%s.htm' %x

    print ('Processed %s' %url)

    dfs = pd.read_html(url)

    df = dfs[0]



    df.columns = cols



    df = df[df['Const. No.'].notnull()]

    df = df.loc[df['Const. No.'].str.isdigit()].reset_index(drop=True)

    df = df.dropna(axis=1,how='all')



    df['Leading Candidate'] = df['Leading Candidate'].str.split('i',expand=True)[0]

    df['Leading Party'] = df['Leading Party'].str.split('iCurrent',expand=True)[0]

    df['Trailing Party'] = df['Trailing Party'].str.split('iCurrent',expand=True)[0]

    df['Trailing Candidate'] = df['Trailing Candidate'].str.split('iAssembly',expand=True)[0]



    results_df27 = results_df27.append(df).reset_index(drop=True)

results_df27 = results_df27.loc[:, :'Status']

results_df27["state Code"]= 'S27'

results_df27["state name"]= 'Jharkhand'



#Uttarakhand S28

url_list = [1]

for x in url_list:

    url = 'https://results.eci.gov.in/pc/en/trends/statewiseS28%s.htm' %x

    print ('Processed %s' %url)

    dfs = pd.read_html(url)

    df = dfs[0]



    df.columns = cols



    df = df[df['Const. No.'].notnull()]

    df = df.loc[df['Const. No.'].str.isdigit()].reset_index(drop=True)

    df = df.dropna(axis=1,how='all')



    df['Leading Candidate'] = df['Leading Candidate'].str.split('i',expand=True)[0]

    df['Leading Party'] = df['Leading Party'].str.split('iCurrent',expand=True)[0]

    df['Trailing Party'] = df['Trailing Party'].str.split('iCurrent',expand=True)[0]

    df['Trailing Candidate'] = df['Trailing Candidate'].str.split('iAssembly',expand=True)[0]



    results_df28 = results_df28.append(df).reset_index(drop=True)

results_df28 = results_df28.loc[:, :'Status']

results_df28["state Code"]= 'S28'

results_df28["state name"]= 'Uttarakhand'



#Telangana S29

url_list = [1,2]

for x in url_list:

    url = 'https://results.eci.gov.in/pc/en/trends/statewiseS29%s.htm' %x

    print ('Processed %s' %url)

    dfs = pd.read_html(url)

    df = dfs[0]



    df.columns = cols



    df = df[df['Const. No.'].notnull()]

    df = df.loc[df['Const. No.'].str.isdigit()].reset_index(drop=True)

    df = df.dropna(axis=1,how='all')



    df['Leading Candidate'] = df['Leading Candidate'].str.split('i',expand=True)[0]

    df['Leading Party'] = df['Leading Party'].str.split('iCurrent',expand=True)[0]

    df['Trailing Party'] = df['Trailing Party'].str.split('iCurrent',expand=True)[0]

    df['Trailing Candidate'] = df['Trailing Candidate'].str.split('iAssembly',expand=True)[0]



    results_df29 = results_df29.append(df).reset_index(drop=True)

results_df29 = results_df29.loc[:, :'Status']

results_df29["state Code"]= 'S29'

results_df29["state name"]= 'Telangana'



#Andaman&nicobarislands U01

url_list = [1]

for x in url_list:

    url = 'https://results.eci.gov.in/pc/en/trends/statewiseU01%s.htm' %x

    print ('Processed %s' %url)

    dfs = pd.read_html(url)

    df = dfs[0]



    df.columns = cols



    df = df[df['Const. No.'].notnull()]

    df = df.loc[df['Const. No.'].str.isdigit()].reset_index(drop=True)

    df = df.dropna(axis=1,how='all')



    df['Leading Candidate'] = df['Leading Candidate'].str.split('i',expand=True)[0]

    df['Leading Party'] = df['Leading Party'].str.split('iCurrent',expand=True)[0]

    df['Trailing Party'] = df['Trailing Party'].str.split('iCurrent',expand=True)[0]

    df['Trailing Candidate'] = df['Trailing Candidate'].str.split('iAssembly',expand=True)[0]



    results_df30 = results_df30.append(df).reset_index(drop=True)

results_df30 = results_df30.loc[:, :'Status']

results_df30["state Code"]= 'U01'

results_df30["state name"]= 'Andaman & Nicobar Islands'



#Chandigarh U02

url_list = [1]

for x in url_list:

    url = 'https://results.eci.gov.in/pc/en/trends/statewiseU02%s.htm' %x

    print ('Processed %s' %url)

    dfs = pd.read_html(url)

    df = dfs[0]



    df.columns = cols



    df = df[df['Const. No.'].notnull()]

    df = df.loc[df['Const. No.'].str.isdigit()].reset_index(drop=True)

    df = df.dropna(axis=1,how='all')



    df['Leading Candidate'] = df['Leading Candidate'].str.split('i',expand=True)[0]

    df['Leading Party'] = df['Leading Party'].str.split('iCurrent',expand=True)[0]

    df['Trailing Party'] = df['Trailing Party'].str.split('iCurrent',expand=True)[0]

    df['Trailing Candidate'] = df['Trailing Candidate'].str.split('iAssembly',expand=True)[0]



    results_df31 = results_df31.append(df).reset_index(drop=True)



results_df31 = results_df31.loc[:, :'Status']

results_df31["state Code"]= 'U02'

results_df31["state name"]= 'Chandigarh'



#Dadra & Nagar Haveli U03

url_list = [1]

for x in url_list:

    url = 'https://results.eci.gov.in/pc/en/trends/statewiseU03%s.htm' %x

    print ('Processed %s' %url)

    dfs = pd.read_html(url)

    df = dfs[0]



    df.columns = cols



    df = df[df['Const. No.'].notnull()]

    df = df.loc[df['Const. No.'].str.isdigit()].reset_index(drop=True)

    df = df.dropna(axis=1,how='all')



    df['Leading Candidate'] = df['Leading Candidate'].str.split('i',expand=True)[0]

    df['Leading Party'] = df['Leading Party'].str.split('iCurrent',expand=True)[0]

    df['Trailing Party'] = df['Trailing Party'].str.split('iCurrent',expand=True)[0]

    df['Trailing Candidate'] = df['Trailing Candidate'].str.split('iAssembly',expand=True)[0]



    results_df32 = results_df32.append(df).reset_index(drop=True)

results_df32 = results_df32.loc[:, :'Status']

results_df32["state Code"]= 'U03'

results_df32["state name"]= 'Dadra & Nagar Haveli'



#Daman & Diu U04

url_list = [1]

for x in url_list:

    url = 'https://results.eci.gov.in/pc/en/trends/statewiseU04%s.htm' %x

    print ('Processed %s' %url)

    dfs = pd.read_html(url)

    df = dfs[0]



    df.columns = cols



    df = df[df['Const. No.'].notnull()]

    df = df.loc[df['Const. No.'].str.isdigit()].reset_index(drop=True)

    df = df.dropna(axis=1,how='all')



    df['Leading Candidate'] = df['Leading Candidate'].str.split('i',expand=True)[0]

    df['Leading Party'] = df['Leading Party'].str.split('iCurrent',expand=True)[0]

    df['Trailing Party'] = df['Trailing Party'].str.split('iCurrent',expand=True)[0]

    df['Trailing Candidate'] = df['Trailing Candidate'].str.split('iAssembly',expand=True)[0]



    results_df33 = results_df33.append(df).reset_index(drop=True)

results_df33 = results_df33.loc[:, :'Status']

results_df33["state Code"]= 'U04'

results_df33["state name"]= 'Daman & Diu'



#Delhi U05

url_list = [1]

for x in url_list:

    url = 'https://results.eci.gov.in/pc/en/trends/statewiseU05%s.htm' %x

    print ('Processed %s' %url)

    dfs = pd.read_html(url)

    df = dfs[0]



    df.columns = cols



    df = df[df['Const. No.'].notnull()]

    df = df.loc[df['Const. No.'].str.isdigit()].reset_index(drop=True)

    df = df.dropna(axis=1,how='all')



    df['Leading Candidate'] = df['Leading Candidate'].str.split('i',expand=True)[0]

    df['Leading Party'] = df['Leading Party'].str.split('iCurrent',expand=True)[0]

    df['Trailing Party'] = df['Trailing Party'].str.split('iCurrent',expand=True)[0]

    df['Trailing Candidate'] = df['Trailing Candidate'].str.split('iAssembly',expand=True)[0]



    results_df34 = results_df34.append(df).reset_index(drop=True)

results_df34 = results_df34.loc[:, :'Status']

results_df34["state Code"]= 'U05'

results_df34["state name"]= 'Delhi'



#Lakshadweep U06

url_list = [1]

for x in url_list:

    url = 'https://results.eci.gov.in/pc/en/trends/statewiseU06%s.htm' %x

    print ('Processed %s' %url)

    dfs = pd.read_html(url)

    df = dfs[0]



    df.columns = cols



    df = df[df['Const. No.'].notnull()]

    df = df.loc[df['Const. No.'].str.isdigit()].reset_index(drop=True)

    df = df.dropna(axis=1,how='all')



    df['Leading Candidate'] = df['Leading Candidate'].str.split('i',expand=True)[0]

    df['Leading Party'] = df['Leading Party'].str.split('iCurrent',expand=True)[0]

    df['Trailing Party'] = df['Trailing Party'].str.split('iCurrent',expand=True)[0]

    df['Trailing Candidate'] = df['Trailing Candidate'].str.split('iAssembly',expand=True)[0]



    results_df35 = results_df35.append(df).reset_index(drop=True)

results_df35 = results_df35.loc[:, :'Status']

results_df35["state Code"]= 'U06'

results_df35["state name"]= 'Lakshadweep'



#Puducherry U07

url_list = [1]

for x in url_list:

    url = 'https://results.eci.gov.in/pc/en/trends/statewiseU07%s.htm' %x

    print ('Processed %s' %url)

    dfs = pd.read_html(url)

    df = dfs[0]



    df.columns = cols



    df = df[df['Const. No.'].notnull()]

    df = df.loc[df['Const. No.'].str.isdigit()].reset_index(drop=True)

    df = df.dropna(axis=1,how='all')



    df['Leading Candidate'] = df['Leading Candidate'].str.split('i',expand=True)[0]

    df['Leading Party'] = df['Leading Party'].str.split('iCurrent',expand=True)[0]

    df['Trailing Party'] = df['Trailing Party'].str.split('iCurrent',expand=True)[0]

    df['Trailing Candidate'] = df['Trailing Candidate'].str.split('iAssembly',expand=True)[0]



    results_df36 = results_df36.append(df).reset_index(drop=True)

results_df36 = results_df36.loc[:, :'Status']

results_df36["state Code"]= 'U07'

results_df36["state name"]= 'Puducherry'



results_df2 = results_df2.append(results_df1).reset_index(drop=True)

results_df3 = results_df3.append(results_df2).reset_index(drop=True)

results_df4 = results_df4.append(results_df3).reset_index(drop=True)

results_df5 = results_df5.append(results_df4).reset_index(drop=True)

results_df6 = results_df6.append(results_df5).reset_index(drop=True)

results_df7 = results_df7.append(results_df6).reset_index(drop=True)

results_df8 = results_df8.append(results_df7).reset_index(drop=True)

results_df9 = results_df9.append(results_df8).reset_index(drop=True)

results_df10 = results_df10.append(results_df9).reset_index(drop=True)

results_df11 = results_df11.append(results_df10).reset_index(drop=True)

results_df12 = results_df12.append(results_df11).reset_index(drop=True)

results_df13 = results_df13.append(results_df12).reset_index(drop=True)

results_df14 = results_df14.append(results_df13).reset_index(drop=True)

results_df15 = results_df15.append(results_df14).reset_index(drop=True)

results_df16 = results_df16.append(results_df15).reset_index(drop=True)

results_df17 = results_df17.append(results_df16).reset_index(drop=True)

results_df18 = results_df18.append(results_df17).reset_index(drop=True)

results_df19 = results_df19.append(results_df18).reset_index(drop=True)

results_df20 = results_df20.append(results_df19).reset_index(drop=True)

results_df21 = results_df21.append(results_df20).reset_index(drop=True)

results_df22 = results_df22.append(results_df21).reset_index(drop=True)

results_df23 = results_df23.append(results_df22).reset_index(drop=True)

results_df24 = results_df24.append(results_df23).reset_index(drop=True)

results_df25 = results_df25.append(results_df24).reset_index(drop=True)

results_df26 = results_df26.append(results_df25).reset_index(drop=True)

results_df27 = results_df27.append(results_df26).reset_index(drop=True)

results_df28 = results_df28.append(results_df27).reset_index(drop=True)

results_df29 = results_df29.append(results_df28).reset_index(drop=True)

results_df30 = results_df30.append(results_df29).reset_index(drop=True)

results_df31 = results_df31.append(results_df30).reset_index(drop=True)

results_df32 = results_df32.append(results_df31).reset_index(drop=True)

results_df33 = results_df33.append(results_df32).reset_index(drop=True)

results_df34 = results_df34.append(results_df33).reset_index(drop=True)

results_df35 = results_df35.append(results_df34).reset_index(drop=True)

results_df36 = results_df36.append(results_df35).reset_index(drop=True)



print(results_df36)

results_df36.to_csv('Consit2019.csv', index=False)



with open("Consit2019.csv",'r') as f:

    with open("Consit2019updated.csv",'w') as f1:

        next(f)

        row = "constituencyName,pcno,leadingCandidate,leadingParty,trailingCandidate,trailingParty,margin,status,stateCode,stateName"

        f1.write(row)# skip header line

        f1.write("\n")

        for line in f:

            f1.write(line)






