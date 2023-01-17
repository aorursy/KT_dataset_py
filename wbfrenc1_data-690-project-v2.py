import pandas as pd

import numpy as np

import os

import matplotlib.pyplot as plt

from scipy import stats
#Identify columns to bring in from each file



fin_data_cols = ['Ticker Symbol','Period Ending','Net Income', 'Total Assets','Total Equity','Total Revenue','Estimated Shares Outstanding']

sec_data_cols = ['Ticker symbol','Security','GICS Sector','CIK']
# Read in data files



fpath = '../input/nyse/'                                                                                  #Path to data files on Kaggle

#fpath = './'                                                                                               #Local Path to data files

fin_data = pd.read_csv(fpath + 'fundamentals.csv',usecols=fin_data_cols)                                   #Read in the financial data limited to the necessary columns 

sec_data = pd.read_csv(fpath + 'securities.csv',usecols=sec_data_cols)                                     #Read in the price data limited to the necessary columns
fin_data.shape
fin_data.info()
fin_data.sample(5)
fin_data.describe(include='all')
sec_data.shape
sec_data.head(5)
sec_data.info()
sec_data.describe(include='all')
# This is used in Jupyter Labs.  Kaggle correctly Identifies Total Assets.  Total Assets is appearing as an 'object' category and contains commas.  Remove the commas and cast as a float.

'''

fin_data['Total Assets'] = fin_data['Total Assets'].str.replace(',','')                                  #remove the ',' with ''

fin_data['Total Assets'] = fin_data['Total Assets'].astype(float)                                        #convert Total Assets to float'''
fin_data.info()
#CIK is a qualitative label and not a number.  Recast it as a string

sec_data ['CIK'] = sec_data['CIK'].apply(str)
sec_data.info()
fin_data.rename(columns={'Ticker Symbol':'symbol','Estimated Shares Outstanding':'shares'},inplace=True)     #rename columns to increase useability/readability
#Define a function to map the Period Ending according to the rule previously described (interestingly Kaggle and Jupter Labs interpret the date data differently and so I need to adjust this function)



def date_map (date,separator):

    date_comps = date.split(separator)                      #splits the date string at the '/' into a dataframe with three columns

    if int(date_comps[1]) < 7 :                             #tests the month to see if it is before or july or after june

        return (str(int(date_comps[0]) -1) )                #if before july returns the prior year

    else:

        return ((date_comps[0]))                            #if after june retunrs the current year



fin_data['year'] = fin_data['Period Ending'].apply(date_map, args=['-'])
fin_data.sample(5)
fin_data.info()
mis_values = []  

symbols_2015 = set(fin_data[fin_data['year'] == '2015'].symbol.unique())                                  #Identify all unique symbols in 2015 

symbols_2014 = set(fin_data[fin_data['year'] == '2014'].symbol.unique())                                  #Identify all unique symbols in 2014

symbols_2013 = set(fin_data[fin_data['year'] == '2013'].symbol.unique())                                  #Identify all unique symbols in 2013



mis_values = (symbols_2015 | symbols_2014 | symbols_2013 ) - (symbols_2015 & symbols_2014 & symbols_2013 )

print (len(mis_values))
fin_data = fin_data[~fin_data['symbol'].isin(mis_values)]
fin_data = fin_data[~(fin_data['year'] < '2013')]
sec_data.rename(columns={'Ticker symbol':'symbol','Security':'name','GICS Sector':'gics'},inplace=True)
sec_data.head(5)
merged_data = fin_data.merge(sec_data,how='inner',on=['symbol'])                                            #merge f_avgp_data and sec_data on symbol
merged_data.head(5)
merged_data.info()
merged_data.groupby(['year','gics']).count()
merged_data['ROE'] = merged_data['Net Income'] / merged_data['Total Equity']            #Calculate ROE 

merged_data['PM'] = merged_data['Net Income'] / merged_data['Total Revenue']            #Calculate Profit Margin 

merged_data['TAT'] = merged_data['Total Revenue'] / merged_data['Total Assets']         #Calculate Total Asset Turnover

merged_data['EM'] = merged_data['Total Assets'] / merged_data['Total Equity']           #Calcualte Equity Multiplier
merged_data.info()
merged_data.head(5)
ind_stats = merged_data.copy()                                                               #copy the data from merged_data into a new dataframe that will house the industry statistics

ind_stats = ind_stats.melt(id_vars=['symbol','Period Ending','Net Income','Total Assets',    #create a new variable called Ratio that is composed of the ROE and ROE component columns.  This will facilitate the create of descriptive statistics using groupby

                                    'Total Equity','Total Revenue','year','name','gics','shares',

                                    'CIK'],var_name='Ratio',value_name='Value')

ind_stats.groupby(['gics','Ratio'])['Value'].describe()                                      #calculate the descriptive statistics
z = np.abs(stats.zscore(merged_data[['ROE','PM','TAT','EM']]))                             #calculate the z-scores of the ratios in the data    
z.shape
threshold = 4                                                                             #define the threshold of an outlier

outliers = np.where(np.abs(z)>threshold)[:][0]                                            #identify the indexes of the outlier companies
merged_data= merged_data.drop(index=outliers)                                             #drop the outliers from the original merged data set
#Recreate the industry statistics data set = ind_stats

ind_stats = merged_data.copy()                                                               #copy the data from merged_data into a new dataframe that will house the industry statistics

ind_stats = ind_stats.melt(id_vars=['symbol','Period Ending','Net Income','Total Assets',    #create a new variable called Ratio that is composed of the ROE and ROE component columns.  This will facilitate the create of descriptive statistics using groupby

                                    'Total Equity','Total Revenue','year','name','gics','shares',

                                    'CIK'],var_name='Ratio',value_name='Value')

ind_stats = ind_stats.groupby(['gics','Ratio'])['Value'].describe()                                      #calculate the descriptive statistics
ind_stats.head(44)
ind_ratio_means = ind_stats.unstack(level=1)['mean'].reset_index()
ind_ratio_means.replace(['Consumer Discretionary','Information Technology','Telecommunications Services'],['Consumer Discretion','IT','Telecom Services'],inplace=True)      #make appropriate industry label names smaller to facilitate graphing
fig, ax = plt.subplots(1,1,figsize=(12,8))                                                       #creates the overall chart 

ind_ratio_means.sort_values('ROE',inplace=True)                                                  #sort the values so that the chart displays from lowest to highest ROE value

ax.bar(ind_ratio_means['gics'],ind_ratio_means['ROE'])                                           #plots the bar chart that corresponds to ROE

ax.set_title('Chart 1: Industry ROE 2013-2015',fontsize=20,fontweight='bold')                    #Set the title, its fontsize, and weight

ax.set_ylabel('ROE',fontsize =15,fontweight='bold')                                              #set the y labels

ax.set_xlabel('Industry Name',fontsize=15,fontweight='bold')                                     #set the x labels                     

ax.set_xticklabels(ind_ratio_means.gics,rotation = 25,fontsize = 13,ha='right')                  #set the rotation and fontsize of the xticks and center them

ax.set_yticklabels(['{:.0%}'.format(x) for x in ax.get_yticks()])                                #format the y tick labels as %

plt.grid()
ax = merged_data.boxplot(column='ROE',by='gics',rot=45,figsize=(13,8))                              #Plot a box plot graph of the underlying company ROE data

ax.set_title('Chart 2: Industry ROE 2013-2015',fontsize=20,fontweight='bold')                       #Set the title, its fontsize, and weight

ax.set_ylabel('ROE',fontsize =15,fontweight='bold')                                                 #Set the y labels

ax.set_xlabel('Industry Name',fontsize=15,fontweight='bold')                                        #Set the x labels     

ax.set_xticklabels(merged_data.gics.unique(),rotation = 25,fontsize = 13,ha='right')                #Set the rotation and fontsize of the xticks and center them

ax.set_yticklabels(['{:.0%}'.format(x) for x in ax.get_yticks()])                                   #format the y tick labels as %
pos = list(range(len(ind_ratio_means['gics'])))                                                                         #create a list of the number of industry groups

width = 0.2                                                                                                             #define the width of the bar graph



fig, ax = plt.subplots(figsize=(18,8))                                                                                  #creates the overall chart

ind_ratio_means.sort_values('ROE',inplace=True)                                                                         #sort the values so they are ordered as the above graph (ROE smallest to largest)

plt.bar(pos, ind_ratio_means['ROE'],width,alpha=0.5,color='red',label=ind_ratio_means['gics'])                          #graph ROE

plt.bar([p+ width for p in pos], ind_ratio_means['PM'],width,alpha=0.5,color='blue',label=ind_ratio_means['gics'])      #graph PM

plt.bar([p+ 2*width for p in pos], ind_ratio_means['TAT'],width,alpha=0.5,color='gold',label=ind_ratio_means['gics'])   #graph TAT

ax.set_xlabel('Industry Group',fontsize=15,fontweight='bold')                                                           #set the x labels

ax.set_ylabel('',fontsize=15,fontweight='bold')                                                                         #set the y labels

ax.set_xticks(range(len(ind_ratio_means.gics)))                                                                         #adjust the x ticks to show all labels

ax.set_xticklabels(ind_ratio_means.gics,rotation=45,ha='center' )                                                       #set the xtick labels as the gics industry groups, center them and angle them

ax.set_title('Chart 3: 2013 - 2015 Industry ROE, PM, TAT', fontsize=20,fontweight='bold')                               #Title the chart

plt.legend(['ROE','PM','TAT','EM'],loc='upper left')                                                                    #Create the legend

plt.grid()                                                                                                              #show the grid

fig, ax = plt.subplots(1,1,figsize=(12,8))                                                          #creates the overall chart 

ind_ratio_means.sort_values('ROE',inplace=True)                                                     #sort the values so that the chart displays from lowest to highest value

ax.bar(ind_ratio_means['gics'],ind_ratio_means['EM'])                                               #plots the bar chart that corresponds to ROE

ax.set_title('Chart 4: Industry Equity Multiplier 2013-2015',fontsize=20,fontweight='bold')         #Set the title, its fontsize, and weight

ax.set_ylabel('EM',fontsize =15,fontweight='bold')                                                  #set the y labels

ax.set_xlabel('Industry Name',fontsize=15,fontweight='bold')                                        #set the x labels                     

ax.set_xticklabels(ind_ratio_means.gics,rotation = 25,fontsize = 13,ha='right')                     #set the rotation and fontsize of the xticks and center them