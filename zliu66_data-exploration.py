import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import linear_model

import matplotlib.pyplot as plt

from sklearn.utils import shuffle

from sklearn.model_selection import GridSearchCV

import seaborn as sns



from sklearn.model_selection import train_test_split

from sklearn import neural_network

from sklearn import preprocessing

from sklearn import svm

from sklearn import metrics

from sklearn import tree



# Input data files are available in the "input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory





%matplotlib inline

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots

plt.rcParams['image.interpolation'] = 'nearest'

plt.rcParams['image.cmap'] = 'gray'



%load_ext autoreload

%autoreload 2



# list all the input file
# Load the training data



data_dir = '../input/'

Realty = pd.read_csv(data_dir + 'train.csv')

Realty['timestamp'] = pd.to_datetime(Realty['timestamp'])

Realty.set_index('id', inplace = True)

Macro = pd.read_csv(data_dir + 'macro.csv')

Macro['timestamp'] = pd.to_datetime(Macro['timestamp'])

Macro.set_index('timestamp', inplace = True)
RealtyMacro = pd.concat([Realty.reset_index(), Macro.loc[Realty['timestamp']].reset_index().drop('timestamp', axis = 1)], axis = 1)
FontSize = 20

FigSize = (20, 10)

fig, RealtyScatter= plt.subplots(figsize=(10, 10))



RealtyScatter.scatter(Realty['full_sq'], Realty['price_doc'])

 

    

RealtyScatter.legend(fontsize = FontSize)

RealtyScatter.set_xlabel('Apartment area', color = 'k', fontsize = FontSize)

RealtyScatter.set_ylabel('Price', color = 'k', fontsize = FontSize)

RealtyScatter.set_title('House Price vs. Apartment Area Scatter Plot', fontsize = FontSize)



RealtyScatter.spines['bottom'].set_color('k')

RealtyScatter.spines['left'].set_color('k')



RealtyScatter.tick_params('x', colors = 'k', labelsize = FontSize)

RealtyScatter.tick_params('y', colors = 'k', labelsize = FontSize)



# RealtyScatter.set_xlim([0, 1000])

# RealtyScatter.set_ylim([-1000, 1000])
fig2, RealtyScatter2= plt.subplots(figsize=(10, 10))



RealtyScatter2.scatter(Realty['life_sq'], Realty['price_doc'])

 

    

RealtyScatter2.legend(fontsize = FontSize)

RealtyScatter2.set_xlabel('Living room area', color = 'k', fontsize = FontSize)

RealtyScatter2.set_ylabel('Price', color = 'k', fontsize = FontSize)

RealtyScatter2.set_title('House Price vs. Leaving Room Area Scatter Plot', fontsize = FontSize)



RealtyScatter2.spines['bottom'].set_color('k')

RealtyScatter2.spines['left'].set_color('k')



RealtyScatter2.tick_params('x', colors = 'k', labelsize = FontSize)

RealtyScatter2.tick_params('y', colors = 'k', labelsize = FontSize)



# RealtyScatter2.set_xlim([0, 200])

# RealtyScatter.set_ylim([-1000, 1000])

fig2, RealtyScatter3= plt.subplots(figsize=(10, 10))



RealtyScatter3.scatter(Realty['floor'], Realty['max_floor'])

RealtyScatter3.legend(fontsize = FontSize)

RealtyScatter3.set_xlabel('Floorr #', color = 'k', fontsize = FontSize)

RealtyScatter3.set_ylabel('Building floor #', color = 'k', fontsize = FontSize)

RealtyScatter3.set_title('Floor # vs. Building Floor #', fontsize = FontSize)



RealtyScatter3.spines['bottom'].set_color('k')

RealtyScatter3.spines['left'].set_color('k')



RealtyScatter3.tick_params('x', colors = 'k', labelsize = FontSize)

RealtyScatter3.tick_params('y', colors = 'k', labelsize = FontSize)



# RealtyScatter2.set_xlim([0, 200])

# RealtyScatter.set_ylim([-1000, 1000]) 
f, ax = plt.subplots(figsize=(10,10))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



corr = Realty.loc[:,['full_sq', 'life_sq', 'floor', 'max_floor', 'build_year', 'num_room', 

                     'kitch_sq', 'state', 'price_doc']].corr()

# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, cmap=cmap,

            square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)



ax.tick_params('x', colors='k', labelsize = 13)

ax.tick_params('y', colors='k', labelsize = 13)
f1, ax1 = plt.subplots(figsize=(10,10))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



corr1 = Realty.loc[:,['area_m', 'raion_popul', 'full_all', 'male_f', 'female_f',

                     'young_all', 'young_female', 'work_all', 'work_male', 'work_female', 'price_doc']].corr()

# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr1, cmap=cmap,

            square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax1)



ax1.tick_params('x', colors='k', labelsize = 13)

ax1.tick_params('y', colors='k', labelsize = 13)
f2, ax2 = plt.subplots(figsize=(10,10))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



corr2 = Realty.loc[:,['children_preschool', 'preschool_quota', 'preschool_education_centers_raion',

                     'children_school', 'school_quota', 'school_education_centers_raion', 

                     'school_education_centers_top_20_raion', 'university_top_20_raion',

                     'additional_education_raion','price_doc']].corr()

# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr2, cmap=cmap,

            square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax2)

ax2.tick_params('both', colors='k',labelsize = 13)
f3, ax3 = plt.subplots(figsize=(10,10))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



corr3 = Realty.loc[:,['sport_objects_raion', 'culture_objects_top_25_raion', 'shopping_centers_raion', 

                      'park_km', 'fitness_km', 'swim_pool_km', 'ice_rink_km','stadium_km', 'basketball_km', 

                      'shopping_centers_km', 'big_church_km','church_synagogue_km', 'mosque_km', 'theater_km',

                      'museum_km', 'exhibition_km', 'catering_km', 'price_doc']].corr()

# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr3, cmap=cmap,

            square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax3)

ax3.tick_params('both', colors='k',labelsize = 13)
f4, ax4 = plt.subplots(figsize=(10,10))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



corr4 = Realty.loc[:,['thermal_power_plant_km', 'incineration_km', 'water_treatment_km', 'incineration_km',

                      'railroad_station_walk_km', 'railroad_station_walk_min', 'railroad_station_avto_km',

                      'railroad_station_avto_min', 'public_transport_station_km', 

                      'public_transport_station_min_walk', 'water_km', 'mkad_km', 'ttk_km', 'sadovoe_km',

                      'bulvar_ring_km', 'kremlin_km', 'price_doc']].corr()

# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr4, cmap=cmap,

            square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax4)

ax4.tick_params('both', colors='k',labelsize = 13)
RealtyYear = Realty.groupby(Realty['timestamp'].apply(lambda x: x.year)).mean()

# RealtyYear = Realty.groupby(Realty['timestamp'].apply(lambda x: x.to_period('Y'))).mean()





fig11, YearPlot= plt.subplots(figsize=(10, 8))

YearPlot.plot(RealtyYear.index, RealtyYear['price_doc'], marker = 'o', markersize = '12', fillstyle='full')





YearPlot.set_xlabel('Year', color = 'k', fontsize = FontSize)

YearPlot.set_ylabel('Price', color = 'k', fontsize = FontSize)

YearPlot.set_title('House Price vs. Year', fontsize = FontSize)



YearPlot.spines['bottom'].set_color('k')

YearPlot.spines['left'].set_color('k')



YearPlot.tick_params('x', colors = 'k', labelsize = FontSize)

YearPlot.tick_params('y', colors = 'k', labelsize = FontSize)



# RealtyScatter2.set_xlim([0, 200])

# RealtyScatter.set_ylim([-1000, 1000])
RealtyMonth = Realty.groupby(Realty['timestamp'].apply(lambda x: x.to_period('M'))).mean()

fig12, MonthPlot= plt.subplots(figsize=(10, 8))

MonthPlot.plot(RealtyMonth.index.to_timestamp(), RealtyMonth['price_doc'],marker = 'o')

MonthPlot.set_xlabel('Year-Month', color = 'k', fontsize = FontSize)

MonthPlot.set_ylabel('Price', color = 'k', fontsize = FontSize)

MonthPlot.set_title('House Price vs. Year-Month', fontsize = FontSize)



MonthPlot.spines['bottom'].set_color('k')

MonthPlot.spines['left'].set_color('k')



MonthPlot.tick_params('x', colors = 'k', labelsize = 13)

MonthPlot.tick_params('y', colors = 'k', labelsize = FontSize)
RealtyDay = Realty.groupby(Realty['timestamp'].apply(lambda x: x.to_period('D'))).mean()

fig13, DayPlot= plt.subplots(figsize=(10, 8))

DayPlot.plot(RealtyDay.index.to_timestamp(), RealtyDay['price_doc'])



DayPlot.set_xlabel('Year-Month-Day', color = 'k', fontsize = FontSize)

DayPlot.set_ylabel('Price', color = 'k', fontsize = FontSize)

DayPlot.set_title('House Price vs. Year-Month-Day', fontsize = FontSize)



DayPlot.spines['bottom'].set_color('k')

DayPlot.spines['left'].set_color('k')



DayPlot.tick_params('x', colors = 'k', labelsize = 13)

DayPlot.tick_params('y', colors = 'k', labelsize = FontSize)
RealtyMonthAvg = Realty.groupby(Realty['timestamp'].apply(lambda x: x.month)).mean()

RealtyMonthStd = Realty.groupby(Realty['timestamp'].apply(lambda x: x.month)).std()

fig14, MonthAPlot= plt.subplots(figsize=(10, 8))

MonthAPlot.plot(RealtyMonthAvg.index, RealtyMonthAvg['price_doc'], marker = 'o', markersize = '12', fillstyle='full')

#MonthAPlot.fill_between(RealtyMonthAvg.index, RealtyMonthAvg['price_doc'] + RealtyMonthStd['price_doc'], 

#                      RealtyMonthAvg['price_doc'] - RealtyMonthStd['price_doc'], facecolor = 'r', 

#                      alpha = 0.4)





MonthAPlot.set_xlabel('Month', color = 'k', fontsize = FontSize)

MonthAPlot.set_ylabel('Price', color = 'k', fontsize = FontSize)

MonthAPlot.set_title('House Price vs. Month', fontsize = FontSize)



MonthAPlot.spines['bottom'].set_color('k')

MonthAPlot.spines['left'].set_color('k')



MonthAPlot.tick_params('x', colors = 'k', labelsize = FontSize)

MonthAPlot.tick_params('y', colors = 'k', labelsize = FontSize)
fig2, RealtyScatter4 = plt.subplots(figsize=(10, 10))





 

RealtyScatter4.scatter(Macro.loc[RealtyDay.index.to_timestamp()]['oil_urals'], RealtyDay['price_doc'] )

    

    

RealtyScatter4.legend(fontsize = FontSize)

RealtyScatter4.set_xlabel('Crude oil price', color = 'k', fontsize = FontSize)

RealtyScatter4.set_ylabel('Average Daily Realty Price', color = 'k', fontsize = FontSize)

RealtyScatter4.set_title('House Price Scatter Plot', fontsize = FontSize)



RealtyScatter4.spines['bottom'].set_color('k')

RealtyScatter4.spines['left'].set_color('k')



RealtyScatter4.tick_params('x', colors = 'k', labelsize = FontSize)

RealtyScatter4.tick_params('y', colors = 'k', labelsize = FontSize)

f5, ax5 = plt.subplots(figsize=(10,10))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



corr5 = RealtyMacro.loc[:,['oil_urals', 'brent', 'gdp_quart', 'gdp_quart_growth', 'gdp_annual', 'gdp_annual_growth',

                           'cpi', 'ppi', 'gdp_deflator', 'balance_trade', 'balance_trade_growth', 'usdrub', 'eurrub',

                           'net_capital_export',  'average_provision_of_build_contract',

                           'average_provision_of_build_contract_moscow', 'price_doc']].corr()

# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr5, cmap=cmap,

            square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax5)

ax5.tick_params('both', colors='k',labelsize = 13)
f6, ax6 = plt.subplots(figsize=(10,10))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



corr6 = RealtyMacro.loc[:,['brent',

                           'rts', 'micex', 'micex_rgbi_tr', 'micex_cbi_tr', 'deposits_value', 'deposits_growth',

                           'deposits_rate', 'mortgage_value', 'mortgage_growth', 'mortgage_rate', 'grp', 'grp_growth', 

                           'income_per_cap', 'real_dispos_income_per_cap_growth',  'salary', 'salary_growth', 'fixed_basket',

                           'price_doc']].corr()

# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr6, cmap=cmap,

            square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax6)

ax6.tick_params('both', colors='k',labelsize = 13)
f7, ax7 = plt.subplots(figsize=(10,10))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



corr7 = RealtyMacro.loc[:,['brent',

                           'retail_trade_turnover', 'retail_trade_turnover_per_cap', 'retail_trade_turnover_growth', 

                           'labor_force', 'unemployment', 'employment', 'invest_fixed_capital_per_cap', 'invest_fixed_assets', 

                           'profitable_enterpr_share', 'unprofitable_enterpr_share', 'share_own_revenues', 'overdue_wages_per_cap',

                           'fin_res_per_cap', 'marriages_per_1000_cap', 'divorce_rate', 'construction_value', 

                           'price_doc']].corr()

# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr7, cmap=cmap,

            square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax7)

ax7.tick_params('both', colors='k',labelsize = 13)
fig21, RealtyScatter21 = plt.subplots(figsize=(10, 10))





 

RealtyScatter21.scatter(Macro['unemployment'], Macro['employment'] )

    

    

RealtyScatter21.set_xlabel('unemployment', color = 'k', fontsize = FontSize)

RealtyScatter21.set_ylabel('employment', color = 'k', fontsize = FontSize)

RealtyScatter21.set_title('unemployment vs. employment', fontsize = FontSize)



RealtyScatter21.spines['bottom'].set_color('k')

RealtyScatter21.spines['left'].set_color('k')



RealtyScatter21.tick_params('x', colors = 'k', labelsize = FontSize)

RealtyScatter21.tick_params('y', colors = 'k', labelsize = FontSize)
fig31, EmploymentPlot= plt.subplots(figsize=(10, 8))

EmploymentPlot.plot(Macro.index, Macro['employment'])







EmploymentPlot.set_xlabel('Time', color = 'k', fontsize = FontSize)

EmploymentPlot.set_ylabel('Employment', color = 'k', fontsize = FontSize)





EmploymentPlot.spines['bottom'].set_color('k')

EmploymentPlot.spines['left'].set_color('k')



EmploymentPlot.tick_params('x', colors = 'k', labelsize = FontSize)

EmploymentPlot.tick_params('y', colors = 'k', labelsize = FontSize)
fig32, UnemploymentPlot= plt.subplots(figsize=(10, 8))

UnemploymentPlot.plot(Macro.index, Macro['unemployment'])







UnemploymentPlot.set_xlabel('Time', color = 'k', fontsize = FontSize)

UnemploymentPlot.set_ylabel('Employment', color = 'k', fontsize = FontSize)





UnemploymentPlot.spines['bottom'].set_color('k')

UnemploymentPlot.spines['left'].set_color('k')



UnemploymentPlot.tick_params('x', colors = 'k', labelsize = FontSize)

UnemploymentPlot.tick_params('y', colors = 'k', labelsize = FontSize)
fig33, LaborForcePlot= plt.subplots(figsize=(10, 8))

LaborForcePlot.plot(Macro.index, Macro['labor_force'])







LaborForcePlot.set_xlabel('Time', color = 'k', fontsize = FontSize)

LaborForcePlot.set_ylabel('Labor force size', color = 'k', fontsize = FontSize)





LaborForcePlot.spines['bottom'].set_color('k')

LaborForcePlot.spines['left'].set_color('k')



LaborForcePlot.tick_params('x', colors = 'k', labelsize = FontSize)

LaborForcePlot.tick_params('y', colors = 'k', labelsize = FontSize)
fig34, CrudeOilPricePlot= plt.subplots(figsize=(10, 8))

CrudeOilPricePlot.plot(Macro.index, Macro['oil_urals'])







CrudeOilPricePlot.set_xlabel('Time', color = 'k', fontsize = FontSize)

CrudeOilPricePlot.set_ylabel('Crude Oil Urals ($/bbl)', color = 'k', fontsize = FontSize)





CrudeOilPricePlot.spines['bottom'].set_color('k')

CrudeOilPricePlot.spines['left'].set_color('k')



CrudeOilPricePlot.tick_params('x', colors = 'k', labelsize = FontSize)

CrudeOilPricePlot.tick_params('y', colors = 'k', labelsize = FontSize)
fig34, CrudeOilPricePlot= plt.subplots(figsize=(10, 8))

CrudeOilPricePlot.plot(Macro.index, Macro['brent'])





CrudeOilPricePlot.set_xlabel('Time', color = 'k', fontsize = FontSize)

CrudeOilPricePlot.set_ylabel('London Brent ($/bbl)', color = 'k', fontsize = FontSize)





CrudeOilPricePlot.spines['bottom'].set_color('k')

CrudeOilPricePlot.spines['left'].set_color('k')



CrudeOilPricePlot.tick_params('x', colors = 'k', labelsize = FontSize)

CrudeOilPricePlot.tick_params('y', colors = 'k', labelsize = FontSize)
fig35, CrudeOilPrice2= plt.subplots(figsize=(10, 8))

GDP_Quarter_Growth =  CrudeOilPrice2.twinx()



CrudeOilPrice2.plot(Macro.index, Macro['brent'], 'r--', label = 'London Brent')

GDP_Quarter_Growth.plot(Macro.index, Macro['gdp_quart_growth'], 'b-', label = 'Real GDP growth quarter')



CrudeOilPrice2.set_xlabel('Time', color = 'k', fontsize = FontSize)

CrudeOilPrice2.set_ylabel('London Brent ($/bbl)', color = 'r', fontsize = FontSize)

GDP_Quarter_Growth.set_ylabel('Real GDP growth quarter', color = 'b', fontsize = FontSize)



CrudeOilPrice2.spines['bottom'].set_color('k')

CrudeOilPrice2.spines['left'].set_color('r')

GDP_Quarter_Growth.spines['right'].set_color('b')



CrudeOilPrice2.tick_params('x', colors = 'k', labelsize = FontSize)

CrudeOilPrice2.tick_params('y', colors = 'r', labelsize = FontSize)

GDP_Quarter_Growth.tick_params('y', colors = 'b', labelsize = FontSize)



lines, labels = CrudeOilPrice2.get_legend_handles_labels()

lines2, labels2 = GDP_Quarter_Growth.get_legend_handles_labels()

CrudeOilPrice2.legend(lines + lines2, labels + labels2, loc=3, fontsize = FontSize)
fig36, CrudeOilPrice3= plt.subplots(figsize=(10, 8))

CPI =  CrudeOilPrice3.twinx()



CrudeOilPrice3.plot(Macro.index, Macro['brent'], 'r--', label = 'London Brent')

CPI.plot(Macro.index, Macro['cpi'], 'b-', label = 'CPI')



CrudeOilPrice3.set_xlabel('Time', color = 'k', fontsize = FontSize)

CrudeOilPrice3.set_ylabel('London Brent ($/bbl)', color = 'r', fontsize = FontSize)

CPI.set_ylabel('CPI', color = 'b', fontsize = FontSize)



CrudeOilPrice3.spines['bottom'].set_color('k')

CrudeOilPrice3.spines['left'].set_color('r')

CPI.spines['right'].set_color('b')



CrudeOilPrice3.tick_params('x', colors = 'k', labelsize = FontSize)

CrudeOilPrice3.tick_params('y', colors = 'r', labelsize = FontSize)

CPI.tick_params('y', colors = 'b', labelsize = FontSize)



lines, labels = CrudeOilPrice3.get_legend_handles_labels()

lines2, labels2 = CPI.get_legend_handles_labels()

CrudeOilPrice3.legend(lines + lines2, labels + labels2, loc=3, fontsize = FontSize)
fig37, CrudeOilPrice4= plt.subplots(figsize=(10, 8))

PPI =  CrudeOilPrice4.twinx()



CrudeOilPrice4.plot(Macro.index, Macro['brent'], 'r--', label = 'London Brent')

PPI.plot(Macro.index, Macro['ppi'], 'b-', label = 'Real GDP growth quarter')



CrudeOilPrice4.set_xlabel('Time', color = 'k', fontsize = FontSize)

CrudeOilPrice4.set_ylabel('London Brent ($/bbl)', color = 'r', fontsize = FontSize)

PPI.set_ylabel('PPI', color = 'b', fontsize = FontSize)



CrudeOilPrice4.spines['bottom'].set_color('k')

CrudeOilPrice4.spines['left'].set_color('r')

GDP_Quarter_Growth.spines['right'].set_color('b')



CrudeOilPrice4.tick_params('x', colors = 'k', labelsize = FontSize)

CrudeOilPrice4.tick_params('y', colors = 'r', labelsize = FontSize)

PPI.tick_params('y', colors = 'b', labelsize = FontSize)



lines, labels = CrudeOilPrice4.get_legend_handles_labels()

lines2, labels2 = PPI.get_legend_handles_labels()

CrudeOilPrice4.legend(lines + lines2, labels + labels2, loc=3, fontsize = FontSize)
fig38, CPIPPI= plt.subplots(figsize=(10, 8))





CPIPPI.plot(Macro.index, Macro['cpi'], 'r--', label = 'CPI')

CPIPPI.plot(Macro.index, Macro['ppi'], 'b-', label = 'PPI')



CPIPPI.set_xlabel('Time', color = 'k', fontsize = FontSize)

CPIPPI.set_ylabel('PI', color = 'r', fontsize = FontSize)





CPIPPI.spines['bottom'].set_color('k')

CPIPPI.spines['left'].set_color('r')





CPIPPI.tick_params('x', colors = 'k', labelsize = FontSize)

CPIPPI.tick_params('y', colors = 'r', labelsize = FontSize)





CPIPPI.legend(loc=2, fontsize = FontSize)
RealtyMonth2 = RealtyMacro.groupby(Realty['timestamp'].apply(lambda x: x.to_period('M'))).median()

fig41, MonthPlot2= plt.subplots(figsize=(10, 8))

Brent =  MonthPlot2.twinx()





MonthPlot2.plot(RealtyMonth2.index.to_timestamp(), RealtyMonth2['price_doc'],'r-', label = 'Realty price')

Brent.plot(Macro.index, Macro['brent'],'b--', label = 'London Brent')

MonthPlot2.set_xlabel('Year-Month', color = 'k', fontsize = FontSize)

MonthPlot2.set_ylabel('Price', color = 'r', fontsize = FontSize)

Brent.set_ylabel('Brent', color = 'b', fontsize = FontSize)



MonthPlot2.set_title('House Price vs. Year-Month', fontsize = FontSize)



MonthPlot2.spines['bottom'].set_color('k')

MonthPlot2.spines['left'].set_color('r')

Brent.spines['right'].set_color('b')



MonthPlot2.tick_params('x', colors = 'k', labelsize = 13)

MonthPlot2.tick_params('y', colors = 'r', labelsize = FontSize)

Brent.tick_params('y', colors = 'b', labelsize = FontSize)





lines, labels = MonthPlot2.get_legend_handles_labels()

lines2, labels2 = Brent.get_legend_handles_labels()

MonthPlot2.legend(lines + lines2, labels + labels2, loc = 3, fontsize = FontSize)
