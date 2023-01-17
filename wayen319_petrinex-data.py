# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
months = ['01','02','03','04','05','06','07','08','09','10','11','12']

filenames = [] 



for month in months:

    filename = '../input/injectionwell/Vol_2018-'+month+'.CSV'

    filenames.append(filename)



dfs = []

for filename in filenames:

    df = pd.read_csv(filename, index_col='ProductionMonth')

    dfs.append(df)
cols = ['ReportingFacilitySubTypeDesc','ReportingFacilityLocation','ActivityID','ProductID','Volume','Hours','ReportingFacilityProvinceState']



nums = [0,1,2,3,4,5,6,7,8,9,10,11]

dfs_new = []



for num in nums:

    df_new = dfs[num][cols]

    dfs_new.append(df_new)

dfs_con=[]



for num in nums:

    df_con = dfs_new[num][(dfs_new[num]['ProductID']=='WATER') & (dfs_new[num]['ActivityID']=='INJ')& (dfs_new[num]['ReportingFacilityProvinceState']=='AB')  & ((dfs_new[num]['ReportingFacilitySubTypeDesc']== 'DISPOSAL') | (dfs_new[num]['ReportingFacilitySubTypeDesc']== 'DISPOSAL (ISSUED BY AER ONLY)') | (dfs_new[num]['ReportingFacilitySubTypeDesc']== 'DISPOSAL (APPROVED AS PART OF A WASTE PLANT)'))]

    

    dfs_con.append(df_con)

import matplotlib.pyplot as plt



month = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

rates = []

volumes = []



for num in nums:

    volume = dfs_con[num]['Volume'].astype(float).sum(axis = 0, skipna = True) /1000000

    print(volume)

    volumes.append(volume)



print(sum(volumes))

    

plt.figure(1, figsize = (8,4))

plt.bar(month, volumes, color="#85bee0")

plt.title('2018 Water Disposal Volume by Month', fontsize=18)

plt.xlabel('Month',fontsize=16)

plt.xticks(rotation='vertical',fontsize=14)

plt.ylabel('$Mm^3$',fontsize=16)

plt.yticks(fontsize=14)

plt.show()



print(dfs_con[4].info())



#color for waste #85e0a2

#color for water #85bee0
X_test = pd.DataFrame(columns=['LSD', 'Section', 'Township', 'Range','Meridian'])

num_data = 0



for num in nums:

    num_data = num_data + dfs_con[num]['ReportingFacilityLocation'].count()



print(num_data)

print(dfs_con[0]['ReportingFacilityLocation'].iloc[1][0:2])



a = 0



for num in nums:

    xx = dfs_con[num]['ReportingFacilityLocation'].count()

    phs = list(range(a,a+xx))

    ph = list(range(0,xx))

    

    A=dfs_con[num]['ReportingFacilityLocation'].iloc[ph].str[0:2].to_frame('LSD')

    B=dfs_con[num]['ReportingFacilityLocation'].iloc[ph].str[3:5].to_frame('Section')

    C=dfs_con[num]['ReportingFacilityLocation'].iloc[ph].str[6:9].to_frame('Township')

    D=dfs_con[num]['ReportingFacilityLocation'].iloc[ph].str[10:12].to_frame('Range')

    E=dfs_con[num]['ReportingFacilityLocation'].iloc[ph].str[13].to_frame('Meridian')

    F = pd.concat([A,B,C,D,E], axis=1)

    

    X_test = pd.concat([X_test,F], axis=0)

    

    a = a + xx

    
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.tree import DecisionTreeRegressor

from sklearn import preprocessing
in_test_fn = '../input/new-test-input/test_input.csv'

out_test_fn = '../input/test-lsd-to-gps/test_output.csv'

X_train = pd.read_csv(in_test_fn)

Y_train = pd.read_csv(out_test_fn)



X_train = X_train.dropna()

X_test = X_test.dropna()

Y_train = Y_train.dropna()



print(X_test.isnull().sum())



X_train = X_train.astype(int) 

X_test = X_test.astype(int)
result = DecisionTreeRegressor().fit(X_train, Y_train).predict(X_test)

df = pd.DataFrame(result, columns =['lat','long'])
import folium

map_disp = folium.Map(location=[55.116460, -115.139502],zoom_start=6, tiles='Stamen Terrain')



print(len(df))



for i in range(0,len(df)):

    folium.CircleMarker([df['lat'].iloc[i], df['long'].iloc[i]], color='crimson', fill='true', radius=4).add_to(map_disp)



map_disp