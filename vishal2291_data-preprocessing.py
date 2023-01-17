# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/rainfall-in-india"))



# Any results you write to the current directory are saved as output.
cropdata = pd.read_csv("../input/crop-production-statistics-from-1997-in-india/apy.csv")

print(cropdata.size)

print(cropdata.head())

print(set(cropdata['State_Name']))

print(cropdata.columns)

print(set(cropdata['Season']))

print(set(cropdata['Crop']))

print(set(cropdata['Crop_Year']))

#print(set(cropdata['District_Name']))

crop_ap = cropdata[cropdata["State_Name"]=="Andhra Pradesh"]

print(crop_ap.head())

crop_ap_2000 = crop_ap[crop_ap["Crop_Year"]==2000]

crop_ap_2000_wholeyear = crop_ap_2000[crop_ap_2000["Season"]=="Kharif     "]

print(crop_ap_2000.head())

print(crop_ap.size)

print(crop_ap_2000.size)

print(crop_ap_2000_wholeyear.size)

print(crop_ap_2000_wholeyear.head())

ananthapur = crop_ap_2000_wholeyear[crop_ap_2000_wholeyear["District_Name"]=="ANANTAPUR"]

ananthapur.plot(kind="bar",x="Crop",y="Production")

crop_ap_2000_wholeyear.plot(kind="bar",x="Crop",y="Production")
raindata = pd.read_csv("../input/rainfall-in-india/rainfall in india 1901-2015.csv")

print(raindata.head())
print(raindata.columns)

print(set(raindata['SUBDIVISION']))

# ap_rain= raindata[raindata['SUBDIVISION']=='Andhra Pradesh']

# print(set(raindata['Year']))

# ap_rain_2000 = ap_rain[ap_rain['Year'] == 2000]

# print(ap_rain_2000)
valid_states_rain = raindata[(raindata['SUBDIVISION']=='BIHAR') | (raindata['SUBDIVISION']=='KERALA') | (raindata['SUBDIVISION']=='ARUNACHAL PRADESH')|(raindata['SUBDIVISION']=='TAMIL NADU')|(raindata['SUBDIVISION']=='JAMMU & KASHMIR')|(raindata['SUBDIVISION']=='UTTARAKHAND')|(raindata['SUBDIVISION']=='ORISSA')|(raindata['SUBDIVISION']=='HIMACHAL PRADESH')|(raindata['SUBDIVISION']=='LAKSHADWEEP')|(raindata['SUBDIVISION']=='PUNJAB')|(raindata['SUBDIVISION']=='CHHATTISGARH')|(raindata['SUBDIVISION']=='ANDAMAN & NICOBAR ISLANDS')|(raindata['SUBDIVISION']=='JHARKHAND')]
print(set(valid_states_rain['SUBDIVISION']))

print(valid_states_rain.head())

valid_states_rain = valid_states_rain[['SUBDIVISION','YEAR','Jun-Sep']]

print(valid_states_rain.head())

print(valid_states_rain.describe())
print(cropdata.head())

print(cropdata.describe())

states_set = set(cropdata['State_Name'])

print(states_set)

year_set = set(cropdata['Crop_Year'])

print(year_set)
crop_processed_data = pd.DataFrame(columns = ['State','Year','Crop','Area','Production'])

print(crop_processed_data)
cropdata = cropdata.drop('Season',axis = 1)

cropdata = cropdata.drop('District_Name',axis = 1)
print(states_set)

k=0

for state in states_set:

    

    tempdf = cropdata[cropdata['State_Name']==state]

    tempdf = tempdf.drop('State_Name',axis = 1)

    #print(tempdf)

    year_set = set(tempdf['Crop_Year'])

    crop_set = set(tempdf['Crop'])

    #     print(year_set)

    #     print(crop_set)

    #     print("--------------------------------------------")

    for year in year_set:

        yeartempdf = tempdf[tempdf['Crop_Year']==year]

        for crop in crop_set:

            area = 0

            production = 0

            for index,row in yeartempdf.iterrows():

                if(row['Crop']==crop):

                    area = area+row['Area']

                    production = production+row['Production']

            crop_processed_data.loc[k] = [state,year,crop,area,production]

            k+=1

    

                    

            

    
crop_processed_data
crop_processed_data.to_csv(r"kharif_dataset.csv",index=False)