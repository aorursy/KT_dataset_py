# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
file = '../input/Indicators.csv'
data = pd.read_csv(file,sep=',')
# List of all SADC countries for use in filtering
countries = ['Lesotho','South Africa','Zimbabwe','Botswana','Swaziland','Namibia','Zambia','Mozambique','Seychelles',
             'Tanzania','Angola','Congo','Malawi','Mauritius','Madagascar','SADC']

# A filtering mask to obtain a data frame of SADC countries
countries_mask = data['CountryName'].isin(countries)
sadc_countries = data[countries_mask] # The SADC countries data frame

# List for stylistic features of the graphs
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
             (230, 10, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
             (188, 189, 34), (219, 229, 141), (23, 190, 207), (158, 210, 229),
             (124,230,54), (34,100,85), (230,164,192), (12,33,122),(223,35,87),
             (111,233,68),(233,233,110),(110,110,43),(31,119,180),(31, 119, 180), (174, 199, 232),
            (255, 187, 120),(255,255,255),(245,245,245),(235,235,20),(225,225,40)]  

# change of the tuples above into RGB colours
for i in range(len(tableau20)):    
    r, g, b = tableau20[i]    
    tableau20[i] = (r / 255., g / 255., b / 255.)

# Function to create a SADC wide average of the indicators to be explored
def sadc_stats(dataFrame):
    SADC = dataFrame.groupby('Year',as_index=False).mean()
    SADC['CountryName']='SADC'
    SADC['CountryCode']='SADC' # Both country name and country code are set as SADC
    SADC['IndicatorName']=dataFrame['IndicatorName'].iloc[0] # Add the indicator being explored to the SADC data frame 
    SADC['IndicatorCode']=dataFrame['IndicatorCode'].iloc[0]
    return SADC #return the new SADC data frame

# Function to print a line graph of the data for a list of countries
# The country names are passed in as a list
def print_data(dataFrame,alist):
    axis = plt.subplot(111)
    for country in alist:
        curr = dataFrame[dataFrame['CountryName']==country]
        if not curr.empty: #check to see if the country has data pertaining to the indicator being explored
            curr.plot(ax=axis,x='Year',y='Value',figsize=(8,6),marker='o',grid=True,label=country)
            axis.set_ylabel(dataFrame['IndicatorName'].iloc[0])
            axis.set_facecolor('k')
    plt.legend(loc='upper left',frameon=True,bbox_to_anchor=(1.05,1))
    plt.title(dataFrame['IndicatorName'].iloc[0])
    plt.show()

# Function to print an area graph of certain countries in a data frame
# The country names are passed in as a list
def print_dataA(dataFrame,alist):
    axis = plt.subplot(111)
    for country in alist:
        curr = dataFrame[dataFrame['CountryName']==country]
        if not curr.empty: #check to see if the country has data
            curr.plot.area(ax=axis,x='Year',y='Value',figsize=(8,6),grid=True,label=country)
            axis.set_ylabel(dataFrame['IndicatorName'].iloc[0])
            axis.set_facecolor('k')
    plt.legend(loc='upper left',frameon=True,bbox_to_anchor=(1.05,1))
    plt.title(dataFrame['IndicatorName'].iloc[0])
    plt.show()

# Function to print all the data of certain countries in a data frame
# The country names are passed in as a list
def printData(dataFrame,alist):
    ax = plt.subplot(111)
    i=0 # variable to help select colours for the line graph
    for country in alist:
        i+=2 # Incrementation of two so as to avoid similar colours
        curr = dataFrame[dataFrame['CountryName']==country]
        if not curr.empty: #check to see if the country has data
            curr.plot(ax=ax,figsize=(8,6),x='Year',y='Value',label=country,marker='o',color = tableau20[i],grid=True)
            ax.set_ylabel(dataFrame['IndicatorName'].iloc[0])
            ax.set_facecolor('k')


    plt.legend(loc='upper left',frameon=True,bbox_to_anchor=(1.05,1))
    plt.title(dataFrame['IndicatorName'].iloc[0])
    plt.show()
#create a data frame of SADC countries filtered with the GDP (current LCU) data
indicator_mask = 'GDP (current LCU'
mask1 = sadc_countries['IndicatorName'].str.startswith(indicator_mask)
gdp_stage = sadc_countries[mask1]
SADC=sadc_stats(gdp_stage)
gdp_stage = gdp_stage.append(SADC)
gdp_stage = gdp_stage.sort_values('Year',ascending=True)
#Print the GDP data
countries = gdp_stage['CountryName'].unique().tolist()
printData(gdp_stage,countries)
# Rank the countries in terms of GDP
high_gdp = gdp_stage.sort_values(['Year','Value'],ascending=False)
high_gdp.head()
#print the data of the highest two countries by GDP (LCU) against the average of the region
h2countries = ['Tanzania','Madagascar','SADC']
print_dataA(gdp_stage,h2countries)
print_data(gdp_stage,h2countries)
high_gdp.head(15)
#plot the data of the lowest two countries by GDP (LCU) against the region Average 
l2countries = ['Seychelles','Zimbabwe','SADC']
print_data(gdp_stage,l2countries)
# Plot the data of the lowest two ranking countries by GDP
l2countries = ['Seychelles','Zimbabwe']
print_dataA(gdp_stage,l2countries)
print_data(gdp_stage,l2countries)
# create a data frame of the country's GDP at market prices
indicator='GDP at market prices (curr'
mask = sadc_countries['IndicatorName'].str.startswith(indicator)
GDP_us = sadc_countries[mask]
SADC = sadc_stats(GDP_us)
GDP_us = GDP_us.append(SADC)
GDP_us = GDP_us.sort_values('Year',ascending=True)
#GDP_us.head(10)
#plot the data
printData(GDP_us,countries)
# Rank the countries by GDP at market price
max_gdp=GDP_us.sort_values(['Year','Value'],ascending=False)#.iloc[:2]
max_gdp.head(15)
# Plot the data of the two highest countries by GDP against the region average
high_gdp_countries = ['South Africa','Angola','SADC']
print_dataA(GDP_us,high_gdp_countries)
print_data(GDP_us,high_gdp_countries)
# Plot the data of the lowest two countries against the region average
low_gdp_countries=['Lesotho','Seychelles','SADC']
print_data(GDP_us,low_gdp_countries)
# Plot the data of the lowest two countries 
low_gdp_countries=['Lesotho','Seychelles']
print_dataA(GDP_us,low_gdp_countries)
print_data(GDP_us,low_gdp_countries)
# create a data frame of the countries using the Trade in Services indicator
indicator = 'Trade in services'
mask = sadc_countries['IndicatorName'].str.startswith(indicator)
trade_in_services = sadc_countries[mask]
SADC = sadc_stats(trade_in_services)
trade_in_services = trade_in_services.append(SADC)
trade_in_services = trade_in_services.sort_values('Year',ascending=True)
#trade_in_services.head()
# Plot the data
printData(trade_in_services,countries)
# Rank the countries by Trade in services
high_trade = trade_in_services.sort_values(['Year','Value'],ascending=False)
high_trade.head(20)
# Plot the highest two country's data against the region average
high_trade = ['Seychelles','Mauritius','SADC']
print_dataA(trade_in_services,high_trade)
print_data(trade_in_services,high_trade)
# plot the lowest two country's data
low_trade = ['South Africa','Zambia','SADC']
print_data(trade_in_services,low_trade)
# Create a data frame with the service imports indicator
indicator='Service imports (BoP'
service_imports = sadc_countries[sadc_countries['IndicatorName'].str.startswith(indicator)]
SADC = sadc_stats(service_imports)
service_imports = service_imports.append(SADC)
service_imports = service_imports.sort_values('Year',ascending=True)
#service_imports.head()
# plot the data
printData(service_imports,countries)
# Rank the countries by service imports
high_imports = service_imports.sort_values(['Year','Value'],ascending=False)
high_imports.head(20)
# plot the highest two countries against the region average
high_serv_imports = ['South Africa','Angola','SADC']
print_dataA(service_imports,high_serv_imports)
print_data(service_imports,high_serv_imports)
# Plot the lowest two countries against the region average
low_serv_imports = ['Lesotho','Seychelles','SADC']
print_data(service_imports,low_serv_imports)
# plot the lowest two countries
low_serv_imports=['Lesotho','Seychelles']
print_dataA(service_imports,low_serv_imports)
print_data(service_imports,low_serv_imports)
# Create a data frame using the service exports indicator
indicator='Service exports (BoP'
mask = sadc_countries['IndicatorName'].str.startswith(indicator)
service_exports = sadc_countries[mask]
SADC = sadc_stats(service_exports)
service_exports = service_exports.append(SADC)
service_exports = service_exports.sort_values('Year',ascending=True)
#service_exports.head(10)
# plot the data
printData(service_exports,countries)
# Rank the data in terms of the indicator
high_export = service_exports.sort_values(['Year','Value'],ascending=False)
high_export.head(20)
# Plot the highest two countries against the region average
high_export = ['South Africa','Mauritius','Tanzania','SADC']
print_dataA(service_exports,high_export)
print_data(service_exports,high_export)
# plot the lowest two countries against the region average
low_export=['Lesotho','Swaziland','SADC']
print_data(service_exports,low_export)
# plot the lowest two countries
low_export = ['Lesotho','Swaziland']
print_dataA(service_exports,low_export)
print_data(service_exports,low_export)
# Create a data frame with the Insurance and financial services indicator
indicator= "Insurance and financial services (% of service imports, BoP)"
indmask= sadc_countries["IndicatorName"].str.startswith(indicator)
finance_stage = sadc_countries[indmask]
SADC = sadc_stats(finance_stage)
finance_stage = finance_stage.append(SADC)
finance_stage = finance_stage.sort_values('Year',ascending=True)
#finance_stage.head(10)
# Plot the data
printData(finance_stage,countries)
# Rank the countries in terms of the indicator
high_fin = finance_stage.sort_values(['Year','Value'],ascending=False)
high_fin.head(20)
# Plot the highest two countries against the region average
high_fin = ['Zambia','Mauritius','SADC']
print_dataA(finance_stage,high_fin)
print_data(finance_stage,high_fin)
# plot the lowest two countries against the region average
low_fin = ['Seychelles','Namibia','SADC']
print_dataA(finance_stage,low_fin)
print_data(finance_stage,low_fin)
# Plot the lowest two countries
low_fin = ['Seychelles','Namibia']
print_dataA(finance_stage,low_fin)
print_data(finance_stage,low_fin)
# Create the data frame using the Insurance and financial services exports indicator
indicator='Insurance and financial services (% of service exports'
mask = sadc_countries['IndicatorName'].str.startswith(indicator)
fin_exports = sadc_countries[mask]
SADC = sadc_stats(fin_exports)
fin_exports = fin_exports.append(SADC)
fin_exports = fin_exports.sort_values('Year',ascending=True)
#fin_exports.head()
printData(fin_exports,countries)
# Rank the countries in terms of Insuarance and financial services exports
high_exp = fin_exports.sort_values(['Year','Value'],ascending=False)
high_exp.head(20)
high_exp=['South Africa','Botswana','Zambia','SADC']
print_dataA(fin_exports,high_exp)
print_data(fin_exports,high_exp)
low_exp = ['Mozambique','Seychelles','SADC']
#print_dataA(fin_exports,low_exp)
print_data(fin_exports,low_exp)
low_exp = ['Mozambique','Seychelles']
print_dataA(fin_exports,low_exp)
print_data(fin_exports,low_exp)
indicator = 'ICT service exports (%'
mask = sadc_countries['IndicatorName'].str.startswith(indicator)
ICT_exports = sadc_countries[mask]
SADC = sadc_stats(ICT_exports)
ICT_exports = ICT_exports.append(SADC)
ICT_exports = ICT_exports.sort_values('Year',ascending=True)
#ICT_exports.head()
printData(ICT_exports,countries)
high_exp = ICT_exports.sort_values(['Year','Value'],ascending=False)
high_exp.head(20)
high_exp = ['Namibia','Swaziland','Mauritius','SADC']
print_dataA(ICT_exports,high_exp)
print_data(ICT_exports,high_exp)
low_exp = ['Angola','Zambia','SADC']
#print_dataA(ICT_exports,low_exp)
print_data(ICT_exports,low_exp)
low_exp = ['Angola','Zambia']
print_dataA(ICT_exports,low_exp)
print_data(ICT_exports,low_exp)
indicator="ICT service exports (B"
indmask = sadc_countries["IndicatorName"].str.startswith(indicator)
ICT_stage = sadc_countries[indmask]
SADC = sadc_stats(ICT_stage)
ICT_stage = ICT_stage.append(SADC)
ICT_stage = ICT_stage.sort_values('Year',ascending=True)
#ICT_stage.head()
printData(ICT_stage,countries)
# Rank the countries in terms of ICT
high_exp = ICT_stage.sort_values(['Year','Value'],ascending=False)
high_exp.head(20)
high_exp = ['South Africa','Mauritius','SADC']
print_dataA(ICT_stage,high_exp)
print_data(ICT_stage,high_exp)
low_exp = ['Lesotho','Zambia','SADC']
#print_dataA(ICT_stage,low_exp)
print_data(ICT_stage,low_exp)
low_exp = ['Lesotho','Zambia']
print_dataA(ICT_stage,low_exp)
print_data(ICT_stage,low_exp)
indicator ='Communications, computer, etc. (% of service exports, BoP)'
mask= sadc_countries['IndicatorName'].str.startswith(indicator)
comp=sadc_countries[mask]
SADC = sadc_stats(comp)
comp = comp.append(SADC)
comp = comp.sort_values('Year',ascending=True)
#comp.head()
printData(comp,countries)
# Rank the countries
high_exp = comp.sort_values(['Year','Value'],ascending=False)
high_exp.head(20)
high_exp = ['Botswana','Swaziland','SADC']
#print_dataA(comp,high_exp)
print_data(comp,high_exp)
low_exp = ['Angola','Zambia','SADC']
print_dataA(comp,low_exp)
print_data(comp,low_exp)
low_exp = ['Angola','Zambia']
print_dataA(comp,low_exp)
print_data(comp,low_exp)
indicator='GDP at market prices (curr'
mask = data['IndicatorName'].str.startswith(indicator)
global_gdp = data[mask]
global_gdp = global_gdp.sort_values(['Year','Value'],ascending=False)
#global_gdp.head(20)
compare = ['United States','China','Japan','Germany','United Kingdom','France','Brazil','Italy','India','Canada']
mask = global_gdp['CountryName'].isin(compare)
comp_set = global_gdp[mask]
mask = sadc_countries['CountryName'].isin(['South Africa'])
sa = sadc_countries[mask]
mask = sa['IndicatorName'].str.startswith(indicator)
sa = sa[mask]
comp_set = comp_set.append(sa)
comp_set = comp_set.sort_values(['Year'],ascending=True)
#comp_set.head(11)
alist=comp_set['CountryName'].unique().tolist()
printData(comp_set,alist)
counts = global_gdp[global_gdp['CountryName'].isin(compare)]
counts[counts['Year']==2014].describe()
counts[counts['Year']==2014].mean()['Value']/sa[sa['Year']==2014]['Value'].iloc[0]
print_dataA(comp_set,['South Africa','Canada'])
print_data(comp_set,['South Africa','Canada'])
counts[(counts['Year']==2014) & (counts['CountryName'].str.startswith('Canada'))]['Value'].iloc[0]/sa[sa['Year']==2014]['Value'].iloc[0]
indicator = 'Trade in services'
comp_set = data[data['CountryName'].isin(alist)]
trade_stage = comp_set[comp_set['IndicatorName'].str.startswith(indicator)]
trade_stage1 = trade_stage.sort_values(['Year','Value'],ascending=False)
trade_stage1.head(11)
printData(trade_stage,alist)
counts = data[data['CountryName'].isin(compare)]
counts_trade = counts[counts['IndicatorName'].str.startswith(indicator)]
counts_trade[counts_trade['Year']==2014].describe()
sa = sadc_countries[sadc_countries['CountryName'].isin(['South Africa'])]
sa_trade = sa[sa['IndicatorName'].str.startswith(indicator)]
sa_trade[(sa_trade['Year']==2014)]['Value'].iloc[0]
indicator='Service exports (BoP'
serv_exp = comp_set[comp_set['IndicatorName'].str.startswith(indicator)]
serv_exp1 = serv_exp.sort_values(['Year','Value'],ascending=False)
serv_exp1.head(11)
printData(serv_exp,alist)
counts_serv = counts[counts['IndicatorName'].str.startswith(indicator)]
counts_serv[counts_serv['Year']==2014].describe()
sa_serv = sa[sa['IndicatorName'].str.startswith(indicator)]
counts_serv[counts_serv['Year']==2014].mean()['Value']/sa_serv[sa_serv['Year']==2014]['Value'].iloc[0]
indicator='Insurance and financial services (% of service exports'
fin_exp = comp_set[comp_set['IndicatorName'].str.startswith(indicator)]
fin_exp1 = fin_exp.sort_values(['Year','Value'],ascending=False)
fin_exp1.head(11)
printData(fin_exp1,alist)
indicator = 'ICT service exports (%'
ICT = comp_set[comp_set['IndicatorName'].str.startswith(indicator)]
ICT1 = ICT.sort_values(['Year','Value'],ascending=False)
ICT1.head(11)
printData(ICT,alist)
indicator ='Communications, computer, etc. (% of service exports, BoP)'
comp = comp_set[comp_set['IndicatorName'].str.startswith(indicator)]
comp1 = comp.sort_values(['Year','Value'],ascending=False)
comp1.head(11)
printData(comp,alist)
