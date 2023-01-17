import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
data = pd.read_csv("../input/datafile (1).csv")
data.head()
crop_production_data = pd.read_csv("../input/datafile (2).csv")
crop_production_data.head()
fig,axs = plt.subplots(figsize=(10,6))

crop_wise_yield = data.groupby(['Crop']).sum()['Yield (Quintal/ Hectare) ']

plt.plot(crop_wise_yield)

crop_wise_production = data.groupby(['Crop']).sum()['Cost of Production (`/Quintal) C2']/10

plt.plot(crop_wise_production)

plt.xticks(rotation ='vertical')

plt.legend()

# cost of production is 10 times as indicated

# this shows maximum yield/hectare is of SUGARCANE

# sugarcane has low cost of production/quintal 
state_crop_yield = data.groupby(['State'])

index = list(state_crop_yield.indices.keys())

state_crop_yield.sum()[['Cost of Production (`/Quintal) C2', 'Yield (Quintal/ Hectare) ']].plot(kind='bar',figsize=(12,7))
recommended_zone = pd.read_csv('../input/datafile (3).csv')
recommended_zone.drop('Unnamed: 4',axis=1,inplace=True)

recommended_zone.dropna(inplace=True)
recommended_zone.info()
recommended_zone.head()
def state1(row):

    if 'Andhra Pradesh' in row['Recommended Zone']:

        return 1

def state2(row):

    if 'Tamil Nadu' in row['Recommended Zone']:

        return 1

def state3(row):

    if 'Gujarat' in row['Recommended Zone']:

        return 1

def state4(row):

    if 'Orissa' in row['Recommended Zone']:

        return 1

def state5(row):

    if 'Punjab' in row['Recommended Zone']:

        return 1

def state6(row):

    if 'Haryana' in row['Recommended Zone']:

        return 1

def state7(row):

    if 'Uttar Pradesh' in row['Recommended Zone']:

        return 1

def state8(row):

    if 'Rajasthan' in row['Recommended Zone']:

        return 1

def state9(row):

    if 'Karnataka' in row['Recommended Zone']:

        return 1

def state10(row):

    if 'Madhya Pradesh' in row['Recommended Zone']:

        return 1

def state11(row):

    if 'West Bengal' in row['Recommended Zone']:

        return 1
recommended_zone['Andhra Pradesh'] = recommended_zone.apply(state1,axis=1)

recommended_zone['Tamil Nadu']=recommended_zone.apply(state2,axis=1)

recommended_zone['Gujarat']=recommended_zone.apply(state3,axis=1)

recommended_zone['Orissa']=recommended_zone.apply(state4,axis=1)

recommended_zone['Punjab']=recommended_zone.apply(state5,axis=1)

recommended_zone['Haryana']=recommended_zone.apply(state6,axis=1)

recommended_zone['Uttar Pradesh']=recommended_zone.apply(state7,axis=1)

recommended_zone['Rajasthan']=recommended_zone.apply(state8,axis=1)

recommended_zone['Karnataka']=recommended_zone.apply(state9,axis=1)

recommended_zone['Madhya Pradesh']=recommended_zone.apply(state10,axis=1)

recommended_zone['West Bangal']=recommended_zone.apply(state11,axis=1)

# Added the eleven states as columns in the dataframe  
recommended_zone.fillna(0).head()
dataframe = recommended_zone.groupby('Crop').sum().plot(kind='bar',figsize=(15,7))

dataframe

# wheat is almost sown in all the mentioned states

# suitable zones for paddy is Orissa and west Bengal
dataframe = pd.DataFrame(recommended_zone.groupby('Season/ duration in days').count().reset_index())

dataframe1 = pd.DataFrame([dataframe.loc[1:27].sum(),dataframe.loc[29:37].sum()])

dataframe1.drop('Season/ duration in days',axis=1,inplace=True)

dataframe1 = dataframe1.assign(Duration = ['100-190','70-100']) 
dataframe1[['Andhra Pradesh', 'Tamil Nadu',

       'Gujarat', 'Orissa', 'Punjab', 'Haryana', 'Uttar Pradesh', 'Rajasthan',

       'Karnataka', 'Madhya Pradesh', 'West Bangal', 'Duration']].plot(x = 'Duration',kind='bar',figsize=(12,7))

# most favorable state for growing crops in 100-190 days is UP and Rajasthan

# for 70-100 days it is Gujarat
dataframe1

# max number of crops are sown for 100-190 days not for 70-100 days
crop_production_data.head()
crop_production_data.columns = ['Crop', 'Production 2006-07', 'Production 2007-08',

       'Production 2008-09', 'Production 2009-10', 'Production 2010-11',

       'Area 2006-07', 'Area 2007-08', 'Area 2008-09', 'Area 2009-10',

       'Area 2010-11', 'Yield 2006-07', 'Yield 2007-08', 'Yield 2008-09',

       'Yield 2009-10', 'Yield 2010-11']
plt.subplots(figsize=(15,6))

plt.scatter(x='Crop',y='Production 2006-07',data = crop_production_data)

plt.xticks(rotation=90)

plt.show()