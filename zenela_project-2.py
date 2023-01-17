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
data = pd.read_csv("/kaggle/input/education-statistics/edstats-csv-zip-32-mb-/EdStatsData.csv")
data_country = pd.read_csv("/kaggle/input/education-statistics/edstats-csv-zip-32-mb-/EdStatsCountry.csv")
#data.shape
data.head()
#data_country.head()

data.info()
data.groupby('Indicator Code').count()
data_country.head()
data_country.info()
countries_income = data_country["Country Code"].unique()
countries_income = pd.DataFrame({"Country_Code" : data_country["Country Code"].unique(), "Name" : data_country["Short Name"], "Region" : data_country["Region"], "Income_group" : data_country["Income Group"]})
countries_income.to_csv("countries_income.csv")
print(countries_income)
data["Country Name"].describe()

data["Indicator Name"].describe()
countries = data["Country Name"].unique()
countries = pd.DataFrame({"Country Name" : data["Country Name"].unique()})
countries.to_csv("countries.csv")
print(countries)
data_plus_country = pd.merge(data,countries_income, left_on='Country Code', right_on='Country_Code')
data_plus_country.head()
data_plus_country.info()
data_column_reduced = data_plus_country.filter(items=['Region', 'Country Name', 'Country Code', 'Income_group', 'Indicator Name', 'Indicator Code', '2010', '2015', '2025','2030'])
data_column_reduced.head()
data_column_reduced.info()
data_column_reduced = data_column_reduced[data_column_reduced.Income_group.notnull()]
data_column_reduced.head()
data_column_reduced.info()
data_column_reduced.Income_group.unique()
incomes_to_keep = ['Upper middle income', 'High income: nonOECD', 'High income: OECD']
data_reduced_income = data_column_reduced[data_column_reduced.Income_group.isin(incomes_to_keep)]
data_reduced_income.head()
data_reduced_income.info()
data_reduced_internet = data_reduced_income.loc[(data_reduced_income['Indicator Code'] == "IT.NET.USER.P2") | (data_reduced_income['Indicator Code'] == "IT.CMP.PCMP.P2")] 
data_reduced_internet.head()
data_reduced_internet.info()
codes_to_keep = ['IT.NET.USER.P2', 'IT.CMP.PCMP.P2']
data_reduced_itc = data_reduced_income[data_reduced_income['Indicator Code'].isin(codes_to_keep)]
data_reduced_itc.head()
data_reduced_itc.info()
data_reduced_itc.groupby('Indicator Code').count()
codes_to_keep = ['SP.POP.TOTL', 
                 'SP.POP.1524.TO.UN', 
                 'SP.POP.1564.TO', 
                 'SP.SEC.TOTL.IN', 
                 'SP.SEC.UTOT.IN',
                 'SP.TER.TOTL.IN']
data_reduced_demo = data_reduced_income[data_reduced_income['Indicator Code'].isin(codes_to_keep)]
data_reduced_demo.head()
data_reduced_demo.groupby('Indicator Code').count()

data_reduced_demo.info()
codes_to_keep = ['BAR.SEC.CMPT.15UP.ZS', 
                 'BAR.TER.CMPT.15UP.ZS',
                 'UIS.NERA.3',
                 'HH.DHS.SCR',
                 'BAR.SEC.CMPT.15UP.ZS',
                 'UIS.EA.3.AG25T99']
data_reduced_edu = data_reduced_income[data_reduced_income['Indicator Code'].isin(codes_to_keep)]
data_reduced_edu.head()
data_reduced_edu.groupby('Indicator Code').count()
data_reduced_edu.info()
codes_to_keep = ['NY.GNP.PCAP.PP.CD', 
                 'SE.XPD.TOTL.GB.ZS',
                 'NY.GDP.PCAP.PP.CD'
                 'UIS.XGDP.4.FSGOV.FDINSTADM.FFD']
data_reduced_eco = data_reduced_income[data_reduced_income['Indicator Code'].isin(codes_to_keep)]
data_reduced_eco.head()
data_reduced_eco.info()
data_reduced_eco.groupby('Indicator Code').count()
data_reduced_internet = data_reduced_internet.drop(['2025', '2030'], axis=1)
data_reduced_internet.head()
data_reduced_internet = data_reduced_internet[data_reduced_internet['2010'].notnull() | 
                                              data_reduced_internet['2015'].notnull()]
data_reduced_internet.head()
data_reduced_internet.info()
data_reduced_internet['Indicator Name'].unique()
data_reduced_internet['2015'] = data_reduced_internet['2015'].round(1)
data_reduced_internet = data_reduced_internet.sort_values(by='2015', ascending=False)
#data_reduced_internet = data_reduced_internet.groupby(data_reduced_internet['Region'])
data_reduced_internet.head()

pivotInternet = data_reduced_internet.pivot_table(index=['Country Name'], 
                                                  columns=['Indicator Name'], 
                                                  values= '2015')
print(pivotInternet)
import matplotlib.pyplot as plt
import seaborn as sns
data_projections1 = data[data['2025'].notnull()]
data_projections1.head()
data_projections1.info()
data_projections1['Country Name'].describe()
data_projections1 = data_projections1.dropna(axis=1)
data_projections1.head()
data_completeness = data.filter(items=['Country Code',
                                       'Indicator Code',
                                       'Indicator Name',
                                       '2000',
                                       '2001',
                                       '2002',
                                       '2003',
                                       '2004',
                                       '2005',
                                       '2006',
                                       '2007',
                                       '2008',
                                       '2009',
                                       '2010',
                                       '2011',
                                       '2012',
                                       '2013',
                                       '2014',
                                       '2015'])
data_completeness.head()
data_completeness = pd.merge(data_completeness,countries_income, left_on='Country Code', right_on='Country_Code')
data_completeness.head()
data_completeness
data_completeness = data_completeness[data_completeness.Income_group.notnull()]
data_completeness.head()
incomes_to_keep = ['Upper middle income', 'High income: nonOECD', 'High income: OECD']
data_completeness = data_completeness[data_completeness.Income_group.isin(incomes_to_keep)]
data_completeness.head()
data_completeness.info()
codes_to_keep = ['IT.CMP.PCMP.P2']
data_itc_NET = data_completeness[data_completeness['Indicator Code'].isin(codes_to_keep)]
data_itc_NET.info()
codes_to_keep = ['IT.NET.USER.P2']
data_itc_PC = data_completeness[data_completeness['Indicator Code'].isin(codes_to_keep)]
data_itc_PC.info()
#'SP.POP.TOTL', 'SP.POP.1524.TO.UN', 'SP.POP.1564.TO', 'SP.SEC.TOTL.IN', 'SP.SEC.UTOT.IN'
codes_to_keep = ['SP.POP.TOTL']
data_demo_total = data_completeness[data_completeness['Indicator Code'].isin(codes_to_keep)]
data_demo_total.info()
#'SP.POP.TOTL', 'SP.POP.1524.TO.UN', 'SP.POP.1564.TO', 'SP.SEC.TOTL.IN', 'SP.SEC.UTOT.IN'
codes_to_keep = ['SP.POP.1524.TO.UN']
data_demo_1524 = data_completeness[data_completeness['Indicator Code'].isin(codes_to_keep)]
data_demo_1524.info()
#'SP.POP.TOTL', 'SP.POP.1524.TO.UN', 'SP.POP.1564.TO', 'SP.SEC.TOTL.IN', 'SP.SEC.UTOT.IN'
codes_to_keep = ['SP.POP.1564.TO']
data_demo_1564= data_completeness[data_completeness['Indicator Code'].isin(codes_to_keep)]
data_demo_1564.info()
#'SP.POP.TOTL', 'SP.POP.1524.TO.UN', 'SP.POP.1564.TO', 'SP.SEC.TOTL.IN', 'SP.SEC.UTOT.IN'
codes_to_keep = ['SP.SEC.TOTL.IN']
data_demo_SEC= data_completeness[data_completeness['Indicator Code'].isin(codes_to_keep)]
data_demo_SEC.info()
#'SP.POP.TOTL', 'SP.POP.1524.TO.UN', 
#'SP.POP.1564.TO', 'SP.SEC.TOTL.IN', 
#'SP.SEC.UTOT.IN', SP.TER.TOTL.IN, 'SL.UEM.TOTL.ZS'
codes_to_keep = ['SP.SEC.UTOT.IN']
data_demo_USEC= data_completeness[data_completeness['Indicator Code'].isin(codes_to_keep)]
data_demo_USEC.info()
#'SP.POP.TOTL', 'SP.POP.1524.TO.UN', 
#'SP.POP.1564.TO', 'SP.SEC.TOTL.IN', 
#'SP.SEC.UTOT.IN', SP.TER.TOTL.IN, 'SL.UEM.TOTL.ZS'
codes_to_keep = ['SP.TER.TOTL.IN']
data_demo_TER= data_completeness[data_completeness['Indicator Code'].isin(codes_to_keep)]
data_demo_TER.info()
#'SP.POP.TOTL', 'SP.POP.1524.TO.UN', 
#'SP.POP.1564.TO', 'SP.SEC.TOTL.IN', 
#'SP.SEC.UTOT.IN', SP.TER.TOTL.IN, 'SL.UEM.TOTL.ZS'
codes_to_keep = ['SL.UEM.TOTL.ZS']
data_demo_UNEMPL= data_completeness[data_completeness['Indicator Code'].isin(codes_to_keep)]
data_demo_UNEMPL.info()
#'BAR.SEC.CMPT.15UP.ZS', 'BAR.TER.CMPT.15UP.ZS','UIS.NERA.3','HH.DHS.SCR','BAR.SEC.CMPT.15UP.ZS','UIS.EA.3.AG25T99'
codes_to_keep = ['BAR.SEC.CMPT.15UP.ZS']
data_edu_SEC15UP = data_completeness[data_completeness['Indicator Code'].isin(codes_to_keep)]
data_edu_SEC15UP.info()
#'BAR.SEC.CMPT.15UP.ZS', 'BAR.TER.CMPT.15UP.ZS','UIS.NERA.3','HH.DHS.SCR','BAR.SEC.CMPT.15UP.ZS','UIS.EA.3.AG25T99'
codes_to_keep = ['BAR.TER.CMPT.15UP.ZS']
data_edu_TER15UP = data_completeness[data_completeness['Indicator Code'].isin(codes_to_keep)]
data_edu_TER15UP.info()
#'BAR.SEC.CMPT.15UP.ZS', 'BAR.TER.CMPT.15UP.ZS','UIS.NERA.3','HH.DHS.SCR','BAR.SEC.CMPT.15UP.ZS','UIS.EA.3.AG25T99'
codes_to_keep = ['UIS.NERA.3']
data_edu_ENROLUSEC = data_completeness[data_completeness['Indicator Code'].isin(codes_to_keep)]
data_edu_ENROLUSEC.info()
#'BAR.SEC.CMPT.15UP.ZS', 'BAR.TER.CMPT.15UP.ZS','UIS.NERA.3','HH.DHS.SCR','BAR.SEC.CMPT.15UP.ZS','UIS.EA.3.AG25T99'
codes_to_keep = ['HH.DHS.SCR']
data_edu_SECCOMPLITION = data_completeness[data_completeness['Indicator Code'].isin(codes_to_keep)]
data_edu_SECCOMPLITION.info()
#'NY.GNP.PCAP.PP.CD', 'SE.XPD.TOTL.GB.ZS','NY.GDP.PCAP.PP.CD', 'UIS.XGDP.4.FSGOV.FDINSTADM.FFD'
codes_to_keep = ['NY.GNP.PCAP.PP.CD']
data_edu_GNI = data_completeness[data_completeness['Indicator Code'].isin(codes_to_keep)]
data_edu_GNI.info()
#'NY.GNP.PCAP.PP.CD', 'SE.XPD.TOTL.GB.ZS','NY.GDP.PCAP.PP.CD', 'UIS.XGDP.4.FSGOV.FDINSTADM.FFD'
codes_to_keep = ['SE.XPD.TOTL.GB.ZS']
data_edu_EXPEND_ED = data_completeness[data_completeness['Indicator Code'].isin(codes_to_keep)]
data_edu_EXPEND_ED.info()
#'NY.GNP.PCAP.PP.CD', 'SE.XPD.TOTL.GB.ZS','NY.GDP.PCAP.PP.CD', 'UIS.XGDP.4.FSGOV.FDINSTADM.FFD'
codes_to_keep = ['NY.GDP.PCAP.PP.CD']
data_edu_GDP = data_completeness[data_completeness['Indicator Code'].isin(codes_to_keep)]
data_edu_GDP.info()
#'NY.GNP.PCAP.PP.CD', 'SE.XPD.TOTL.GB.ZS','NY.GDP.PCAP.PP.CD','UIS.XGDP.4.FSGOV.FDINSTADM.FFD'
codes_to_keep = ['UIS.XGDP.4.FSGOV.FDINSTADM.FFD']
data_edu_EXPEND_TER = data_completeness[data_completeness['Indicator Code'].isin(codes_to_keep)]
data_edu_EXPEND_TER.info()
data_column_red = data_plus_country.filter(items=['Region',
                                                  'Country Name', 
                                                  'Country Code', 
                                                  'Income_group', 
                                                  'Indicator Name', 
                                                  'Indicator Code', 
                                                  '2010', 
                                                  '2015'])
data_column_red.head()
data_column_red = data_column_red[data_column_red.Income_group.notnull()]
data_column_red.head()
incomes_to_keep = ['Upper middle income', 'High income: nonOECD', 'High income: OECD']
data_middlehighincome = data_column_red[data_column_red.Income_group.isin(incomes_to_keep)]
data_middlehighincome.head()
data_middlehighincome.info()
data_middlehighincome.notna().sum()/len(data_middlehighincome)
indicators_to_keep = ['IT.NET.USER.P2', 
                      'SP.POP.1564.TO', 
                      'SL.UEM.TOTL.ZS', 
                      'BAR.SEC.CMPT.15UP.ZS', 
                      'BAR.TER.CMPT.15UP.ZS', 
                      'NY.GNP.PCAP.PP.CD', 
                      'SE.XPD.TOTL.GB.ZS']
data_shortlist_high = data_middlehighincome[data_middlehighincome['Indicator Code'].isin(indicators_to_keep)]
data_shortlist_high.head()
pivotShortlist_2015 = data_shortlist_high.pivot_table(index=['Country Name'], 
                                                  columns=['Indicator Code'], 
                                                  values= '2015')
pivotShortlist_2015.head()
pivotShortlist_2015 = pivotShortlist_2015.drop(columns=['SE.XPD.TOTL.GB.ZS'])
pivotShortlist_2015.head()
countries_income
indicators_to_keep = [ 'BAR.SEC.CMPT.15UP.ZS', 
                       'BAR.TER.CMPT.15UP.ZS',
                       'SE.XPD.TOTL.GB.ZS']
data_EDU = data_middlehighincome[data_middlehighincome['Indicator Code'].isin(indicators_to_keep)]
data_EDU.head()
pivotShortlist_2010 = data_EDU.pivot_table(index=['Country Name'], 
                                                  columns=['Indicator Code'], 
                                                  values= '2010')
pivotShortlist_2010.head()
pivotShortlist_2010['BAR.SECTERCMPT.15UP'] = pivotShortlist_2010['BAR.SEC.CMPT.15UP.ZS'] + pivotShortlist_2010['BAR.TER.CMPT.15UP.ZS']
pivotShortlist_2010.head()
pivotShortlist= pd.merge(pivotShortlist_2015,pivotShortlist_2010, on='Country Name')
pivotShortlist.head()
pivot_plus_country = pd.merge(pivotShortlist,countries_income, left_on='Country Name', right_on='Name')
pivot_plus_country = pivot_plus_country.sort_values(by=['Region'])
#'East Asia & Pacific', 'Europe & Central Asia',
#       'Latin America & Caribbean', 'Middle East & North Africa',
#      'North America', 'South Asia', 'Sub-Saharan Africa'

country_to_keep = ['East Asia & Pacific', 'Europe & Central Asia', 
                   'Latin America & Caribbean', 'Middle East & North Africa', 
                   'North America', 'Sub-Saharan Africa']
pivot_plus_country = pivot_plus_country[pivot_plus_country['Region'].isin(country_to_keep)]
pivot_plus_country.head()

pivot_plus_country.Region.unique()
pivot_plus_country = pivot_plus_country.round({'IT.NET.USER.P2': 2, 'SE.XPD.TOTL.GB.ZS': 2})
pivot_plus_country.head()
import matplotlib.pyplot as plt
import seaborn as sns
pivot_plus_country.groupby('Region').mean()
plt.figure()
sns.boxplot(data=pivot_plus_country, x="IT.NET.USER.P2", y="Region")
plt.suptitle("Internet users (%)", size=14, y=1)
plt.show()
plt.savefig('Box_internetusers.png')
plt.figure()
sns.boxplot(data=pivot_plus_country, x="SP.POP.1564.TO", y="Region")
plt.xscale('log')
plt.suptitle("Population, ages 15-64", size=14, y=1)
plt.show()
plt.savefig('Box_population.png')
plt.figure()
sns.boxplot(data=pivot_plus_country, x="NY.GNP.PCAP.PP.CD", y="Region")
plt.suptitle("GNI per capita, PPP (current international $)", size=14, y=1)
plt.show()
plt.savefig('Box_GNI.png')
plt.figure()
sns.boxplot(data=pivot_plus_country, x="SL.UEM.TOTL.ZS", y="Region")
plt.suptitle("Unemployment (% of labor force)", size=14, y=1)
plt.show()
plt.savefig('Box_unemployment.png')
plt.figure()
sns.boxplot(data=pivot_plus_country, x="BAR.SEC.CMPT.15UP.ZS", y="Region")
plt.suptitle("Age 15+ Completed Secondary (%)", size=14, y=1)
plt.show()
plt.savefig('Box_seccomplete.png')
plt.figure()
sns.boxplot(data=pivot_plus_country, x="BAR.TER.CMPT.15UP.ZS", y="Region")
plt.suptitle("Age 15+ Completed Tertiary (%)", size=14, y=1)
plt.show()
plt.savefig('Box_tercomplete.png')
plt.figure()
sns.boxplot(data=pivot_plus_country, x="SE.XPD.TOTL.GB.ZS", y="Region")
#plt.title("Expenditure on education as % of total government expenditure (%)")
plt.suptitle("Expenditure on education (% of total expenditure)", size=14, y=1)
plt.show()
plt.savefig('Box_expenditure.png')
sns.set(font_scale=1.5)
plt.figure(figsize=(16,12))
plt.subplot(2,2,1)
sns.boxplot(data=pivot_plus_country, x="IT.NET.USER.P2", y="Region")
plt.xlabel("")
plt.title("Internet users (%)")

plt.subplot(2,2,2)
sns.boxplot(data=pivot_plus_country, x="SP.POP.1564.TO", y="Region")
plt.xscale('log')
plt.xlabel("")
plt.title("Population, ages 15-64, total")
plt.gca().axes.get_yaxis().set_visible(False)

plt.subplot(2,2,3)
sns.boxplot(data=pivot_plus_country, x="BAR.SEC.CMPT.15UP.ZS", y="Region")
plt.xlabel("")
plt.title("Age 15+ Completed Secondary (%)")


plt.subplot(2,2,4)
sns.boxplot(data=pivot_plus_country, x="BAR.TER.CMPT.15UP.ZS", y="Region")
plt.xlabel("")
plt.title("Age 15+ Completed Tertiary (%)")
plt.gca().axes.get_yaxis().set_visible(False)
plt.show()
plt.savefig('Box_internet_schooling.png')

sns.set(font_scale=1.7)
plt.figure(figsize=(20,5))

plt.subplot(1,3,1)
sns.boxplot(data=pivot_plus_country, x="NY.GNP.PCAP.PP.CD", y="Region")
plt.xlabel("")
plt.title("GNI per capita, PPP (int USD)")

plt.subplot(1,3,2)
sns.boxplot(data=pivot_plus_country, x="SE.XPD.TOTL.GB.ZS", y="Region")
plt.xlabel("")
plt.title("Expenditure on education (%)")
plt.gca().axes.get_yaxis().set_visible(False)

plt.subplot(1,3,3)
sns.boxplot(data=pivot_plus_country, x="SL.UEM.TOTL.ZS", y="Region")
plt.xlabel("")
plt.title("Unemployment (%)")
plt.gca().axes.get_yaxis().set_visible(False)
plt.show()
plt.savefig('Box_GNI_expenditure_unemployment.png')

#'East Asia & Pacific', 'Europe & Central Asia',
#       'Latin America & Caribbean', 'Middle East & North Africa',
#      'North America', 'South Asia', 'Sub-Saharan Africa'

country_to_keep = ['East Asia & Pacific']
pivot_EastAsia = pivot_plus_country[pivot_plus_country['Region'].isin(country_to_keep)]
pivot_EastAsia.head()
sns.set(font_scale=1.2)
plt.figure(figsize=(10,10))
sns.scatterplot(data=pivot_EastAsia, 
                y="IT.NET.USER.P2", 
                x="NY.GNP.PCAP.PP.CD", 
                size='BAR.SEC.CMPT.15UP.ZS',
                hue="SL.UEM.TOTL.ZS", 
                sizes=(25,200))
plt.title("East Asia: NET vs GNI by education and unemployment")
plt.show()
plt.savefig('EastAsia_net_GNI_edu_uemp.png')
pivot_EastAsia = pivot_EastAsia.sort_values(by=['IT.NET.USER.P2'], ascending=False)
pivot_EastAsia_itc = pivot_EastAsia[(pivot_EastAsia['IT.NET.USER.P2'] > 70)]

sns.set(font_scale=1)
plt.figure(figsize=(8,1.4))
sns.barplot(data=pivot_EastAsia_itc, y="Name", x="IT.NET.USER.P2")
plt.title("East Asia : Internet users (%) > 70%")
plt.show()
plt.savefig('bar_internetusers_EastAsia.png')

pivot_EastAsia_gni = pivot_EastAsia_itc.sort_values(by=['NY.GNP.PCAP.PP.CD'], ascending=False)
pivot_EastAsia_gni = pivot_EastAsia_gni[(pivot_EastAsia_gni['NY.GNP.PCAP.PP.CD'] > 35000)]
plt.figure(figsize=(8,1.2))
sns.barplot(data=pivot_EastAsia_gni, y="Name", x="NY.GNP.PCAP.PP.CD")
plt.title("East Asia : GNI per capita, PPP (current international USD) > 35kUSD")
plt.show()
plt.savefig('bar_GNI_EastAsia.png')

pivot_EastAsia_edu = pivot_EastAsia_gni.sort_values(by=['BAR.SEC.CMPT.15UP.ZS'], ascending=False)
pivot_EastAsia_edu = pivot_EastAsia_gni[(pivot_EastAsia_gni['BAR.SEC.CMPT.15UP.ZS'] > 40)]
plt.figure(figsize=(8,0.5))
sns.barplot(data=pivot_EastAsia_edu, y="Name", x="BAR.SEC.CMPT.15UP.ZS")
plt.title("East Asia : Age 15+ Completed Secondary (%) > 50")
plt.show()
plt.savefig('bar_SECCOM_EastAsia.png')
#'East Asia & Pacific', 'Europe & Central Asia',
#       'Latin America & Caribbean', 'Middle East & North Africa',
#      'North America', 'South Asia', 'Sub-Saharan Africa'

country_to_keep = ['Europe & Central Asia']
pivot_EuropeCenAsia = pivot_plus_country[pivot_plus_country['Region'].isin(country_to_keep)]
pivot_EuropeCenAsia.head()
sns.set(font_scale=1.2)
plt.figure(figsize=(10,10))
sns.scatterplot(data=pivot_EuropeCenAsia, 
                y="IT.NET.USER.P2", 
                x="NY.GNP.PCAP.PP.CD", 
                size='BAR.SEC.CMPT.15UP.ZS',
                hue="SL.UEM.TOTL.ZS", 
                sizes=(20,200))
plt.suptitle("Europe and Central Asia: NET vs GNI by education and unemployment", size=14, y=0.92)
plt.show()
plt.savefig('EuropeCenAsia_net_GNI_edu_uemp.png')
pivot_EuropeCenAsia = pivot_EuropeCenAsia.sort_values(by=['IT.NET.USER.P2'], ascending=False)
pivot_EuropeCenAsia_itc = pivot_EuropeCenAsia[(pivot_EuropeCenAsia['IT.NET.USER.P2'] > 70)]
pivot_EuropeCenAsia_gni = pivot_EuropeCenAsia_itc.sort_values(by=['NY.GNP.PCAP.PP.CD'], ascending=False)
pivot_EuropeCenAsia_gni = pivot_EuropeCenAsia_gni[(pivot_EuropeCenAsia_gni['NY.GNP.PCAP.PP.CD'] > 35000)]
pivot_EuropeCenAsia_edu = pivot_EuropeCenAsia_gni.sort_values(by=['BAR.SEC.CMPT.15UP.ZS'], ascending=False)
pivot_EuropeCenAsia_edu = pivot_EuropeCenAsia_edu[(pivot_EuropeCenAsia_edu['BAR.SEC.CMPT.15UP.ZS'] > 40)]

sns.set(font_scale=1.6)
plt.figure(figsize=(45,15))
plt.suptitle('Europe and Central Asia', size=50, y=1)
plt.subplot(1,3,1)
sns.barplot(data=pivot_EuropeCenAsia_itc, y="Name", x="IT.NET.USER.P2")
plt.xlabel('')
plt.ylabel('')
plt.title("Internet users (%) > 70%", fontsize=26)

plt.subplot(1,3,2)
sns.barplot(data=pivot_EuropeCenAsia_gni, y="Name", x="NY.GNP.PCAP.PP.CD")
plt.xlabel('')
plt.ylabel('')
plt.title("GNI per capita, PPP (current international USD) > 35kUSD", fontsize=26)

plt.subplot(1,3,3)
sns.barplot(data=pivot_EuropeCenAsia_edu, y="Name", x="BAR.SEC.CMPT.15UP.ZS")
plt.xlabel('')
plt.ylabel('')
plt.title("Age 15+ Completed Secondary (%) > 40%", fontsize=26)
plt.show()
plt.savefig('bar_EUCentralAsia.png')





#'East Asia & Pacific', 'Europe & Central Asia',
#       'Latin America & Caribbean', 'Middle East & North Africa',
#      'North America', 'South Asia', 'Sub-Saharan Africa'

country_to_keep = ['Latin America & Caribbean']
pivot_LatAmCar = pivot_plus_country[pivot_plus_country['Region'].isin(country_to_keep)]
pivot_LatAmCar.head()
sns.set(font_scale=1.2)
plt.figure(figsize=(10,10))
sns.scatterplot(data=pivot_LatAmCar, 
                y="IT.NET.USER.P2", 
                x="NY.GNP.PCAP.PP.CD", 
                size='BAR.SEC.CMPT.15UP.ZS',
                hue="SL.UEM.TOTL.ZS", 
                sizes=(20,200))
plt.suptitle("Latin America Caribbean: NET vs GNI by education and unemployment", size=14, y=0.92)
plt.show()
plt.savefig('LatAmCar_net_GNI_edu_uemp.png')
pivot_LatAmCar = pivot_LatAmCar.sort_values(by=['IT.NET.USER.P2'], ascending=False)
pivot_LatAmCar = pivot_LatAmCar[(pivot_LatAmCar['IT.NET.USER.P2'] > 70)]
sns.set(font_scale=1)
plt.figure(figsize=(8,0.7))
sns.barplot(data=pivot_LatAmCar, y="Name", x="IT.NET.USER.P2")
plt.title("Latin America & Caribbean: Internet users (%) > 70%")
plt.show()
plt.savefig('bar_internetusers_LatAmCar.png')

#pivot_LatAmCar = pivot_LatAmCar.sort_values(by=['NY.GNP.PCAP.PP.CD'], ascending=False)
#pivot_LatAmCar = pivot_LatAmCar[(pivot_LatAmCar['NY.GNP.PCAP.PP.CD'] > 35000)]
#plt.figure(figsize=(8,0.7))
#sns.barplot(data=pivot_LatAmCar, y="Name", x="NY.GNP.PCAP.PP.CD")
#plt.suptitle("Latin America & Caribbean: GNI per capita, PPP (current international USD) > 35kUSD", size=14, y=1.3)
#plt.show()
#plt.savefig('bar_GNI_LatAmCar.png')
#'East Asia & Pacific', 'Europe & Central Asia',
#       'Latin America & Caribbean', 'Middle East & North Africa',
#      'North America', 'South Asia', 'Sub-Saharan Africa'

country_to_keep = ['Middle East & North Africa']
pivot_MiddleEastNorthAfrica = pivot_plus_country[pivot_plus_country['Region'].isin(country_to_keep)]
pivot_MiddleEastNorthAfrica.head()
pivot_MiddleEastNorthAfrica_itc = pivot_MiddleEastNorthAfrica.sort_values(by=['IT.NET.USER.P2'], ascending=False)
pivot_MiddleEastNorthAfrica_itc = pivot_MiddleEastNorthAfrica_itc[(pivot_MiddleEastNorthAfrica_itc['IT.NET.USER.P2'] > 70)]
plt.figure(figsize=(8,1.5))
sns.barplot(data=pivot_MiddleEastNorthAfrica_itc, y="Name", x="IT.NET.USER.P2")
plt.title("Middle East & North Africa: Internet users (%) > 70%")
plt.show()
plt.savefig('bar_ITC_MiddleEastNorthAfrica.png')

pivot_MiddleEastNorthAfrica_gni = pivot_MiddleEastNorthAfrica_itc.sort_values(by=['NY.GNP.PCAP.PP.CD'], ascending=False)
pivot_MiddleEastNorthAfrica_gni = pivot_MiddleEastNorthAfrica_gni[(pivot_MiddleEastNorthAfrica_gni['NY.GNP.PCAP.PP.CD'] > 35000)]
plt.figure(figsize=(8,1))
sns.barplot(data=pivot_MiddleEastNorthAfrica_gni, y="Name", x="NY.GNP.PCAP.PP.CD")
plt.title("Middle East & North Africa: GNI per capita, PPP (current international USD) > 35kUSD")
plt.show()
plt.savefig('bar_GNI_MiddleEastNorthAfrica.png')

#pivot_MiddleEastNorthAfrica_edu = pivot_MiddleEastNorthAfrica_gni.sort_values(by=['BAR.SEC.CMPT.15UP.ZS'], ascending=False)
#pivot_MiddleEastNorthAfrica_edu = pivot_MiddleEastNorthAfrica_edu[(pivot_MiddleEastNorthAfrica_edu['BAR.SEC.CMPT.15UP.ZS'] > 35000)]
#plt.figure(figsize=(8,2))
#sns.barplot(data=pivot_MiddleEastNorthAfrica_edu, y="Name", x="BAR.SEC.CMPT.15UP.ZS")
#plt.title("Middle East & North Africa: Age 15+ Completed Secondary (%) > 40%")
#plt.show()
#plt.savefig('bar_EDU_MiddleEastNorthAfrica.png')
#'East Asia & Pacific', 'Europe & Central Asia',
#       'Latin America & Caribbean', 'Middle East & North Africa',
#      'North America', 'South Asia', 'Sub-Saharan Africa'

country_to_keep = ['North America']
pivot_NorthAm = pivot_plus_country[pivot_plus_country['Region'].isin(country_to_keep)]
pivot_NorthAm.head()
pivot_NorthAm = pivot_NorthAm.sort_values(by=['IT.NET.USER.P2'], ascending=False)
pivot_NorthAm_itc = pivot_NorthAm[(pivot_NorthAm['IT.NET.USER.P2'] > 70)]
plt.figure(figsize=(8,0.8))
sns.barplot(data=pivot_NorthAm_itc, y="Name", x="IT.NET.USER.P2")
plt.title("North America: Internet users (%) > 70%")
plt.show()
plt.savefig('bar_itc_NorthAm.png')

pivot_NorthAm_gni = pivot_NorthAm_itc.sort_values(by=['NY.GNP.PCAP.PP.CD'], ascending=False)
pivot_NorthAm_gni = pivot_NorthAm_gni[(pivot_NorthAm_gni['NY.GNP.PCAP.PP.CD'] > 35000)]
plt.figure(figsize=(8,0.7))
sns.barplot(data=pivot_NorthAm_gni, y="Name", x="NY.GNP.PCAP.PP.CD")
plt.title("North America: GNI per capita, PPP (current international USD) > 35kUSD ")
plt.show()
plt.savefig('bar_GNI_NorthAm.png')

#pivot_NorthAm_edu = pivot_NorthAm_gni.sort_values(by=['BAR.SEC.CMPT.15UP.ZS'], ascending=False)
#pivot_NorthAm_edu = pivot_NorthAm_edu[(pivot_NorthAm_edu['BAR.SEC.CMPT.15UP.ZS'] > 40)]
#plt.figure(figsize=(8,0.8))
#sns.barplot(data=pivot_NorthAm_edu, y="Name", x="BAR.SEC.CMPT.15UP.ZS")
#plt.title("North America: Age 15+ Completed Secondary (%) > 40% ")
#plt.show()
#plt.savefig('bar_GNI_NorthAm.png')



pivot_NorthAm = pivot_NorthAm.sort_values(by=['NY.GNP.PCAP.PP.CD'], ascending=False)
pivot_NorthAm = pivot_NorthAm[(pivot_NorthAm['NY.GNP.PCAP.PP.CD'] > 35000)]
plt.figure(figsize=(8,0.8))
sns.barplot(data=pivot_NorthAm, y="Name", x="NY.GNP.PCAP.PP.CD")
plt.suptitle("North America: GNI per capita, PPP (current international $) > 35k$ ", size=14, y=1.2)
plt.show()
plt.savefig('bar_GNI_NorthAm.png')
#'East Asia & Pacific', 'Europe & Central Asia',
#       'Latin America & Caribbean', 'Middle East & North Africa',
#      'North America', 'South Asia', 'Sub-Saharan Africa'

country_to_keep = ['Sub-Saharan Africa']
pivot_SubAfr = pivot_plus_country[pivot_plus_country['Region'].isin(country_to_keep)]
pivot_SubAfr.head()
pivot_SubAfr = pivot_SubAfr.sort_values(by=['IT.NET.USER.P2'], ascending=False)
#pivot_SubAfr = pivot_SubAfr[(pivot_SubAfr['IT.NET.USER.P2'] > 70)]
plt.figure(figsize=(8,2))
sns.barplot(data=pivot_SubAfr, y="Name", x="IT.NET.USER.P2")
plt.suptitle("Sub-Saharan Africa: Internet users (%) > 70%", size=14, y=1.1)
plt.show()
#plt.savefig('bar_internetusers_SSAfrica.png')
#pivot_SubAfr = pivot_SubAfr.sort_values(by=['NY.GNP.PCAP.PP.CD'], ascending=False)
#plt.figure(figsize=(8,2))
#sns.barplot(data=pivot_SubAfr, y="Name", x="NY.GNP.PCAP.PP.CD")
#plt.suptitle("Sub-Saharan Africa: GNI per capita, PPP (current international $)", size=14, y=1.1)
#plt.show()
#plt.savefig('bar_GNI_SSAfrica.png')
sns.set(font_scale=1.7)
plt.figure(figsize=(10,10))
sns.scatterplot(data=pivot_plus_country,
                y="IT.NET.USER.P2", 
                x="NY.GNP.PCAP.PP.CD", 
                size='BAR.SEC.CMPT.15UP.ZS',
                hue="Income_group", 
                sizes=(20,200))
plt.suptitle("NET vs GNI by Income", size=14, y=0.92)
plt.show()
plt.savefig('NETGNI_income.png')
sns.set(font_scale=1.7)
plt.figure(figsize=(15,15));
sns.scatterplot(data=pivot_plus_country,
                x="NY.GNP.PCAP.PP.CD",
                y="IT.NET.USER.P2", 
                size='BAR.SEC.CMPT.15UP.ZS',
                hue="Region",
                sizes=(40,350));
plt.ylabel('Internet users (%)')
plt.xlabel('GNI per capita, PPP (int USD)')
plt.suptitle("NET vs GNI and Secondary Education by Region", size=20, y=0.91);
plt.show();
plt.savefig('NETGNI.png');
data_itc_PC = data_itc_PC.round(2)
data_itc_PC.head()

data_itc_PC=data_itc_PC.groupby(data_itc_PC['Region']).data_itc_PC['2015'].agg(max,min,mean)
print(data_itc_PC)
pivot_EuropeCenAsia_edu.head()
pivot_EuropeCenAsia_edu.Name.unique()
Countries_to_keep = ['Germany',
                     'Switzerland',
                     'Sweden', 
                     'United Kingdom',
                     'Austria',
                     'Norway']
data_itc_PC_EU = data_itc_PC[data_itc_PC['Name'].isin(Countries_to_keep)]
data_itc_PC_EU = data_itc_PC_EU.drop(['Country Code', 'Indicator Code', 'Indicator Name', 'Country_Code', 'Region', 'Income_group'], axis=1)
data_itc_PC_EU = data_itc_PC_EU.rename(columns={'Name':'Country'})
data_itc_PC_EU.head()
data_itc_PC_EU = data_itc_PC_EU.set_index('Country')
data_itc_PC_EU.head()
data_itc_PC_years_EU = data_itc_PC_EU.transpose()
data_itc_PC_years_EU = data_itc_PC_years_EU.round(2)
data_itc_PC_years_EU.head()
pivot_EuropeCenAsia_gni.head()
pivot_EuropeCenAsia_gni.Name.unique()
Countries_to_keep = ['Germany',
                     'Switzerland',
                     'Sweden', 
                     'United Kingdom',
                     'Austria',
                     'Norway']
data_itc_PC_EU = data_itc_PC[data_itc_PC['Name'].isin(Countries_to_keep)]
data_itc_PC_EU = data_itc_PC_EU.drop(['Country Code', 'Indicator Code', 'Indicator Name', 'Country_Code', 'Region', 'Income_group'], axis=1)
data_itc_PC_EU = data_itc_PC_EU.rename(columns={'Name':'Country'})
data_itc_PC_EU.head()
sns.set(font_scale=1.5)
plt.figure(figsize=(20,10))
for country in Countries_to_keep:
    sns.lineplot(data=data_itc_PC_years_EU,
                 x=data_itc_PC_years_EU.index,
                 y=country,
                 label=country)
plt.ylabel('Internet usage %', fontsize=25);

data_demo_UNEMPL.head()
Countries_to_keep = ['Czech Republic',
                     'Kazakhstan',
                     'Slovak Republic',
                     'Slovenia', 
                     'Hungary',
                     'Lithuania',
                     'Estonia',
                     'Latvia',
                     'Italy']
data_unempl_EU = data_demo_UNEMPL[data_demo_UNEMPL['Name'].isin(Countries_to_keep)]
data_unempl_EU = data_unempl_EU.drop(['Country Code', 'Indicator Code', 'Indicator Name', 'Country_Code', 'Region', 'Income_group'], axis=1)
data_unempl_EU = data_unempl_EU.rename(columns={'Name':'Country'})
data_unempl_EU.head()
data_unempl_EU = data_unempl_EU.set_index('Country')
data_unempl_EU.head()
data_unempl_years_EU = data_unempl_EU.transpose()
data_unempl_years_EU = data_unempl_years_EU.round(2)
data_unempl_years_EU.head()
sns.set(font_scale=1.5)
plt.figure(figsize=(20,10))
for country in Countries_to_keep:
    sns.lineplot(data=data_unempl_years_EU,
                 x=data_unempl_years_EU.index,
                 y=country,
                 label=country)
plt.ylabel('Unemployment %', fontsize=25);
Countries_to_keep = ['Australia',
                     'New Zealand',
                     'Singapore',
                     'United States', 
                     'Canada',
                     'South Africa']
data_itc_PROM = data_itc_PC[data_itc_PC['Name'].isin(Countries_to_keep)]
data_itc_PROM = data_itc_PROM.drop(['Country Code', 'Indicator Code', 'Indicator Name', 'Country_Code', 'Region', 'Income_group'], axis=1)
data_itc_PROM = data_itc_PROM.rename(columns={'Name':'Country'})
data_itc_PROM = data_itc_PROM.set_index('Country')
data_itc_years_PROM = data_itc_PROM.transpose()
data_itc_years_PROM = data_itc_years_PROM.round(2)
data_itc_years_PROM.head()

data_unempl_PROM = data_demo_UNEMPL[data_demo_UNEMPL['Name'].isin(Countries_to_keep)]
data_unempl_PROM = data_unempl_PROM.drop(['Country Code', 'Indicator Code', 'Indicator Name', 'Country_Code', 'Region', 'Income_group'], axis=1)
data_unempl_PROM = data_unempl_PROM.rename(columns={'Name':'Country'})
data_unempl_PROM = data_unempl_PROM.set_index('Country')
data_unempl_years_PROM = data_unempl_PROM.transpose()
data_unempl_years_PROM = data_unempl_years_PROM.round(2)
data_unempl_years_PROM.head()
sns.set(font_scale=1.5)
plt.figure(figsize=(30,10))
plt.subplot(1,2,1)
for country in Countries_to_keep:
    sns.lineplot(data=data_itc_years_PROM,
                 x=data_itc_years_PROM.index,
                 y=country,
                 label=country)
plt.ylabel('Internet usage %');


plt.subplot(1,2,2)
for country in Countries_to_keep:
    sns.lineplot(data=data_unempl_years_PROM,
                 x=data_unempl_years_PROM.index,
                 y=country,
                 label=country)
plt.ylabel('Unemployment %');
data_projections = data_plus_country.filter(items=['Region',
                                                   'Country Name',
                                                   'Country Code',
                                                   'Income_group',
                                                   'Indicator Name',
                                                   'Indicator Code',
                                                   '2010',
                                                   '2015',
                                                   '2020',
                                                   '2025',
                                                   '2030',
                                                   '2035',
                                                   '2040',
                                                   '2045',
                                                   '2050',
                                                   '2055',
                                                   '2060',
                                                   '2065',
                                                   '2070'])

data_projections.head()
indicators_to_keep = ['PRJ.ATT.2064.2.MF'                    ,
                      'PRJ.ATT.2064.3.MF', 
                      'PRJ.ATT.2064.4.MF']
data_wprojections = data_projections[data_projections['Indicator Code'].isin(indicators_to_keep)]
data_wprojections = data_wprojections[data_wprojections.Income_group.notnull()]
incomes_to_keep = ['Upper middle income',
                   'High income: nonOECD',
                   'High income: OECD']
data_wprojections = data_wprojections[data_wprojections['Income_group'].isin(incomes_to_keep)]
Countries_to_keep = ['Australia',
                     'New Zealand',
                     'Singapore',  
                     'United States', 
                     'Canada',
                     'South Africa']
data_wprojections = data_wprojections[data_wprojections['Country Name'].isin(Countries_to_keep)]
indicators_to_keep = ['PRJ.ATT.2064.2.MF']
data_wprojections_lsec = data_wprojections[data_wprojections['Indicator Code'].isin(indicators_to_keep)]
data_wprojections_lsec = data_wprojections_lsec.drop(['Indicator Code', 'Country Code', 'Indicator Name', 'Region', 'Income_group'], axis=1)
data_wprojections_lsec = data_wprojections_lsec.rename(columns={'Country Name':'Country'})
data_wprojections_lsec = data_wprojections_lsec.set_index('Country')
data_wprojections_lsec = data_wprojections_lsec.transpose()
data_wprojections_lsec.head()





indicators_to_keep = ['PRJ.ATT.2064.3.MF']
data_wprojections_usec = data_wprojections[data_wprojections['Indicator Code'].isin(indicators_to_keep)]
data_wprojections_usec = data_wprojections_usec.drop(['Indicator Code', 'Country Code', 'Indicator Name', 'Region', 'Income_group'], axis=1)
data_wprojections_usec = data_wprojections_usec.rename(columns={'Country Name':'Country'})
data_wprojections_usec = data_wprojections_usec.set_index('Country')
data_wprojections_usec = data_wprojections_usec.transpose()
data_wprojections_usec.head()
indicators_to_keep = ['PRJ.ATT.2064.4.MF']
data_wprojections_ter = data_wprojections[data_wprojections['Indicator Code'].isin(indicators_to_keep)]
data_wprojections_ter = data_wprojections_ter.drop(['Indicator Code', 'Country Code', 'Indicator Name', 'Region', 'Income_group'], axis=1)
data_wprojections_ter = data_wprojections_ter.rename(columns={'Country Name':'Country'})
data_wprojections_ter = data_wprojections_ter.set_index('Country')
data_wprojections_ter = data_wprojections_ter.transpose()
data_wprojections_ter.head()
sns.set(font_scale=3.2)
plt.figure(figsize=(50,35))
plt.subplot(2,2,1)
for country in Countries_to_keep:
    sns.lineplot(data=data_wprojections_lsec,
                 x=data_wprojections_lsec.index,
                 y=country,
                 label=country)
plt.ylabel('Lower Secondary max attained, age 20-64 (%)');

plt.subplot(2,2,2)
for country in Countries_to_keep:
    sns.lineplot(data=data_wprojections_usec,
                 x=data_wprojections_usec.index,
                 y=country,
                 label=country)
plt.ylabel('Upper Secondary max attained, age 20-64 (%)');

plt.subplot(2,2,3)
for country in Countries_to_keep:
    sns.lineplot(data=data_wprojections_ter,
                 x=data_wprojections_ter.index,
                 y=country,
                 label=country)
plt.ylabel('Tertiary max attained, age 20-64 (%)');

