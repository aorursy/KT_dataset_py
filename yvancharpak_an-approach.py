#The classics

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd # para leer datos

import sklearn.manifold

import os

import random

import glob



from sklearn.preprocessing import StandardScaler

import sklearn.metrics

import sklearn.cluster



import torch
folderset = [folder for folder in glob.glob("/kaggle/input/uncover/UNCOVER/" + "**/")]

mobility = ["/kaggle/input/uncover/UNCOVER/geotab/","/kaggle/input/uncover/UNCOVER/google_mobility/", "/kaggle/input/uncover/UNCOVER/un_world_food_programme/"]

ressources = ["/kaggle/input/uncover/UNCOVER/hifld/hifld/"]

healthcareData = ["/kaggle/input/uncover/UNCOVER/oecd/","/kaggle/input/uncover/UNCOVER/us_cdc/us_cdc/"]

covidData = ["/kaggle/input/uncover/UNCOVER/worldometer/","/kaggle/input/uncover/UNCOVER/HDE/","/kaggle/input/uncover/UNCOVER/johns_hopkins_csse/","/kaggle/input/uncover/UNCOVER/world_bank/","/kaggle/input/uncover/UNCOVER/ECDC/","/kaggle/input/uncover/UNCOVER/WHO/","/kaggle/input/uncover/UNCOVER/our_world_in_data/"]

OtherFacilities = ["/kaggle/input/uncover/UNCOVER/OpenTable/"]



csv_Mobility = []

csv_ressources = []

csv_health = []

csv_covid = []

csv_otherFacilities = []
#Mobility

print("\n Mobility \n")

num_carpetas = len(mobility)

for i in range(num_carpetas):

    folder_here = mobility[i] + '*'

    for name in glob.glob(folder_here):

        print(name)

        csv_Mobility.append(name)

#ressources

print("\n Ressources \n")

num_carpetas = len(ressources)

for i in range(num_carpetas):

    folder_here = ressources[i] + '*'

    for name in glob.glob(folder_here):

        print(name)

        csv_ressources.append(name)

#Healthcare

print("\n Healthcare Data \n")

num_carpetas = len(healthcareData)

for i in range(num_carpetas):

    folder_here = healthcareData[i] + '*'

    for name in glob.glob(folder_here):

        print(name)

        csv_health.append(name)

#Covid 

print("\n Covid \n")

num_carpetas = len(covidData)

for i in range(num_carpetas):

    folder_here = covidData[i] + '*'

    for name in glob.glob(folder_here):

        print(name)

        csv_covid.append(name)

#OtherFacilities

print("\n OtherFacilities \n")

num_carpetas = len(OtherFacilities)

for i in range(num_carpetas):

    folder_here = OtherFacilities[i] + '*'

    for name in glob.glob(folder_here):

        print(name)

        csv_otherFacilities.append(name)

                

#We realize that the the school file should be in OtherFacilities



csv_otherFacilities.append(csv_covid.pop(3))



print(csv_otherFacilities)
#HealthCare Utilization

WorldHC = pd.read_csv("/kaggle/input/uncover/UNCOVER/oecd/health-care-utilization.csv")

colombianHC = WorldHC[WorldHC['country'] == 'Colombia']

italianHC = WorldHC[WorldHC['country'] == 'Italy']

japanHC = WorldHC[WorldHC['country'] == 'Japan']

variables=WorldHC['var'].unique()

Mundo = pd.DataFrame(columns= variables)

ColombiaHC = pd.DataFrame(columns= variables)

ItaliaHC = pd.DataFrame(columns= variables)

JapanHC = pd.DataFrame(columns= variables)

#WorldAveg, Colombian and Italian

for i in range(9):

    anio = 2010+i

    year = WorldHC[WorldHC['year'] == anio]

    by_var = year.groupby('var')

    new_row = pd.Series(data=by_var.median()['value'], name=anio)

    Mundo = Mundo.append(new_row)

    

    yearC = colombianHC[colombianHC['year'] == anio]

    by_varC = yearC.groupby('var')

    new_rowC = pd.Series(data=by_varC.median()['value'], name=anio)

    ColombiaHC = ColombiaHC.append(new_rowC)

    

    yearI = italianHC[italianHC['year'] == anio]

    by_varI = yearI.groupby('var')

    new_rowI = pd.Series(data=by_varI.median()['value'], name=anio)

    ItaliaHC = ItaliaHC.append(new_rowI)

    

    yearJ = japanHC[japanHC['year'] == anio]

    by_varJ = yearJ.groupby('var')

    new_rowJ = pd.Series(data=by_varJ.median()['value'], name=anio)

    JapanHC = JapanHC.append(new_rowJ)

print(np.shape(Mundo),np.shape(ColombiaHC),np.shape(ItaliaHC ), np.shape(JapanHC))



plt.figure(figsize=(18, 8))

anios = list(Mundo.index) 

for i in range(3):

    graficaremos =['ACATHEPB', 'ACATIMMU', 'CONSCOVI']

    plt.subplot(1,3,i+1)

    plt.plot(anios,ColombiaHC[graficaremos[i]], label='Colombia')

    plt.plot(anios,ItaliaHC[graficaremos[i]], label='Italia')

    plt.plot(anios,JapanHC[graficaremos[i]], label='Japan')

    plt.legend()

    plt.xlabel('Year')

    plt.ylabel('Coverage')

    plt.title('Evolution of the coverage on {}'.format(graficaremos[i]))



#Mobility

WorldM = pd.read_csv('/kaggle/input/uncover/UNCOVER/google_mobility/regional-mobility.csv')

movs = list(WorldM.keys())[3::]

colombianM = WorldM[WorldM['country'] == 'Colombia']

colombianM = colombianM[colombianM['region'] == 'Total']

italianM = WorldM[WorldM['country'] == 'Italy']

italianM = italianM[italianM['region'] == 'Total']

japanM = WorldM[WorldM['country'] == 'Japan']

japanM = japanM[japanM['region'] == 'Total']

colombianM = colombianM.set_index("date", drop = False)

italianM = italianM.set_index("date", drop = False)

japanM = japanM.set_index("date", drop = False)

colombianM = colombianM[movs]

italianM = italianM[movs]

japanM = japanM[movs]

print(np.shape(colombianM),np.shape(italianM),np.shape(japanM))



plt.figure(figsize=(18, 15))

Fechas = list(colombianM.index) 

for i in range(3):

    graficaremos = random.choice(movs)

    plt.subplot(3,1,i+1)

    plt.plot(Fechas,colombianM[graficaremos], label='Colombia')

    plt.plot(Fechas,italianM[graficaremos], label='Italia')

    plt.plot(Fechas,japanM[graficaremos], label='Japan')

    plt.legend()

    plt.xlabel('Day')

    plt.xticks(Fechas[::14])

    plt.ylabel('Mobility (%)')

    plt.title('Evolution of the mobility to {}'.format(graficaremos))


#COVID-19 

WorldTests = pd.read_csv("/kaggle/input/uncover/UNCOVER/our_world_in_data/covid-19-testing-all-observations.csv")

WorldCases2 = pd.read_csv("/kaggle/input/uncover/UNCOVER/our_world_in_data/coronavirus-disease-covid-19-statistics-and-research.csv")

#As we have different starting dates for each country we have to merge the resulting dataframes to be able to obtain a correct display of the datas. 

#Tests

dats = ['cumulative_total','cumulative_total_per_thousand', 'daily_change_in_cumulative_total_per_thousand',]

colombianT = WorldTests[WorldTests['entity'] == 'Colombia - samples processed']

italianT = WorldTests[WorldTests['entity'] == 'Italy - tests performed']

japanT = WorldTests[WorldTests['entity'] == 'Japan - tests performed']

colombianT = colombianT.set_index("date", drop = False)

italianT = italianT.set_index("date", drop = False)

japanT = japanT.set_index("date", drop = False)

colombianT = colombianT[dats]

italianT = italianT[dats]

japanT = japanT[dats]



Tests = japanT.join(italianT.join(colombianT, lsuffix='_ITA', rsuffix='_COL'), lsuffix='_JPN',  rsuffix='')

dats1 = list(map(lambda x: str(x)+'_COL', dats))

dats2 = list(map(lambda x: str(x)+'_ITA', dats))

#dats3 = list(map(lambda x: str(x)+'_JPN', dats))

colombianTest = Tests[dats1]

italianTest = Tests[dats2]

japanTest = Tests[dats]

print(np.shape(colombianTest),np.shape(italianTest),np.shape(japanTest) )



#Cases and Deaths



dat = ['total_cases','total_deaths', 'total_cases_per_million', 'total_deaths_per_million','new_cases_per_million','new_deaths_per_million']

colombianCD2 = WorldCases2[WorldCases2['iso_code'] == 'COL']

italianCD2 = WorldCases2[WorldCases2['iso_code'] == 'ITA']

japanCD2 = WorldCases2[WorldCases2['iso_code'] == 'JPN']

colombianCD2 = colombianCD2.set_index("date", drop = False)

italianCD2 = italianCD2.set_index("date", drop = False)

japanCD2 = japanCD2.set_index("date", drop = False)



colombianCD2 = colombianCD2[dat]

italianCD2 = italianCD2[dat]

japanCD2 = japanCD2[dat]



COVID = japanCD2.join(italianCD2.join(colombianCD2, lsuffix='_ITA', rsuffix='_COL'),lsuffix='_JPN', rsuffix = '')

dat1 = list(map(lambda x: str(x)+'_COL', dat))

dat2 = list(map(lambda x: str(x)+'_ITA', dat))

#dat3 = list(map(lambda x: str(x)+'_JPN', dat))

colombianCOVID = COVID[dat1]

italianCOVID = COVID[dat2]

japanCOVID = COVID[dat]

print(np.shape(colombianCOVID),np.shape(italianCOVID), np.shape(japanCOVID) )



plt.figure(figsize=(18, 20))

plt.subplot(4,1,1)

FechasT = list(Tests.index)

plt.plot(FechasT,Tests['cumulative_total_per_thousand_COL'], label='Colombia')

plt.plot(FechasT,Tests['cumulative_total_per_thousand_ITA'], label='Italia')

plt.plot(FechasT,Tests['cumulative_total_per_thousand'], label='Japan')

plt.legend()

plt.xlabel('Day')

plt.xticks(FechasT[::14])

plt.ylabel('Tests per Thousand')

plt.title('Evolution of the COVID tests performed')

          

plt.subplot(4,1,2)

plt.plot(FechasT,Tests['cumulative_total_per_thousand_COL'], label='Colombia')

plt.plot(FechasT,Tests['cumulative_total_per_thousand_ITA'], label='Italia')

plt.plot(FechasT,Tests['cumulative_total_per_thousand'], label='Japan')

plt.legend()

plt.xlabel('Day')

plt.xticks(FechasT[::14])

plt.ylabel('Tests per Thousand')

plt.title('Evolution of the COVID tests performed')

       

plt.subplot(4,1,3)

Fechas = list(COVID.index)

plt.plot(Fechas,COVID['total_cases_COL'], label='Colombia')

plt.plot(Fechas,COVID['total_cases_ITA'], label='Italia')

plt.plot(Fechas,COVID['total_cases'], label='Japan')

plt.legend()

plt.xlabel('Day')

plt.xticks(Fechas[::14])

plt.ylabel('Cases')

plt.title('Evolution of the COVID cases')     

    

plt.subplot(4,1,4)

plt.plot(Fechas,COVID['total_deaths_COL'], label='Colombia')

plt.plot(Fechas,COVID['total_deaths_ITA'], label='Italia')

plt.plot(Fechas,COVID['total_deaths'], label='Japan')

plt.legend()

plt.xlabel('Day')

plt.xticks(Fechas[::14])

plt.ylabel('Deaths')

plt.title('Evolution of the COVID deaths')

#This allows to select the health indicators we have for all three countries. 

ColombiaHC1 = ColombiaHC.drop(2018, axis=0)

ColombiaHC1 = ColombiaHC1.dropna(axis = 1)

Colombs = list(ColombiaHC1.keys())

ItaliaHC1 = ItaliaHC[Colombs]

ItaliaHC1 = ItaliaHC1.drop(2018, axis=0)

ItaliaHC1 = ItaliaHC1.dropna(axis = 1)

JapanHC1 = JapanHC[Colombs]

JapanHC1 = JapanHC1.drop(2018, axis=0)

JapanHC1 = JapanHC1.dropna(axis = 1)

Japans = list(JapanHC1.keys())

ColombiaHC1 = ColombiaHC1[Japans]

ItaliaHC1 = ItaliaHC1[Japans]

#Now we unite the COVID and Mobility matrixes

Colombia = colombianCOVID.join(colombianM)

Italy = italianCOVID.join(italianM)

Japan = japanCOVID.join(japanM)

#We create the health care columns.  

japonsito = pd.DataFrame(columns= Japans)

colombito = pd.DataFrame(columns= Japans)

italianito = pd.DataFrame(columns= Japans)

diasAna = list(Colombia.index)

for i in range(len(diasAna)):

    new_rowJ = pd.Series(data=JapanHC1.loc[2017], name=diasAna[i])

    japonsito = japonsito.append(new_rowJ)

    new_rowC =pd.Series(data=ColombiaHC1.loc[2017], name=diasAna[i])

    colombito = colombito.append(new_rowC)

    new_rowI =pd.Series(data=ItaliaHC1.loc[2017], name=diasAna[i])

    italianito= italianito.append(new_rowI)

#We add the health care columns.

Colombia = Colombia.join(colombito)

Italy = Italy.join(italianito)

Japan = Japan.join(japonsito)

#We replace the nans.

Colombia = Colombia.replace(np.nan,0)

Italy = Italy.replace(np.nan,0)

Japan = Japan.replace(np.nan,0)

#We take what is the NCPM

Y_1_CO = np.asarray(Colombia.pop('new_cases_per_million_COL'))

Y_1_IT = np.asarray(Italy.pop('new_cases_per_million_ITA'))

Y_1_JP = np.asarray(Japan.pop('new_cases_per_million'))

#We take what is TDPM

Y_2_CO = np.asarray(Colombia.pop('new_deaths_per_million_COL'))

Y_2_IT = np.asarray(Italy.pop('new_deaths_per_million_ITA'))

Y_2_JP = np.asarray(Japan.pop('new_deaths_per_million'))

#We noramalize our data.

scaler = sklearn.preprocessing.StandardScaler()

Colombia = scaler.fit_transform(Colombia)

Italy = scaler.fit_transform(Italy)

Japan = scaler.fit_transform(Japan)
model1 = torch.nn.Sequential(

    torch.nn.Linear(14, 25),

    torch.nn.Linear(25, 35),

    torch.nn.Linear(35, 10),

    torch.nn.Linear(10, 1)

)



#model1 = torch.nn.Sequential(

#    torch.nn.Conv1d(1, 10, kernel_size=8, stride=1),

#    torch.nn.Conv1d(10, 6, kernel_size=4, stride=1),

#    torch.nn.Conv1d(6, 3, kernel_size=1, stride=1),

#    torch.nn.Conv1d(3, 1, kernel_size=2, stride=3)

#)



model2 = torch.nn.Sequential(

    torch.nn.Conv1d(1, 10, kernel_size=8, stride=1),

    torch.nn.Conv1d(10, 6, kernel_size=4, stride=1),

    torch.nn.Conv1d(6, 3, kernel_size=1, stride=1),

    torch.nn.Conv1d(3, 1, kernel_size=2, stride=3)

)





distance1 = torch.nn.KLDivLoss()

optimizer1 = torch.optim.Adam(model1.parameters(), lr=9E-5, weight_decay=1E-3)

optimizer1.zero_grad()



distance2 = torch.nn.KLDivLoss()

optimizer2 = torch.optim.Adam(model2.parameters(), lr=1E-4, weight_decay=1E-3)

optimizer2.zero_grad()

epochs = 650

for epoch in range(epochs):

    #Training Italy

    X_new1 = np.expand_dims(Italy, 1) 

    inputs1 = torch.autograd.Variable(torch.Tensor(X_new1).float())

    targets1 = torch.autograd.Variable(torch.Tensor(Y_1_IT).float())

    

    

    out1 = model1(inputs1)

    out1 = out1.squeeze(dim=1) # necesario para quitar la dimension intermedia de channel

    loss1 = distance1(out1, targets1)

    loss1.backward()

    optimizer1.step()

    

    #Training Japan

    X_new2 = np.expand_dims(Japan, 1) 

    inputs2 = torch.autograd.Variable(torch.Tensor(X_new2).float())

    targets2 = torch.autograd.Variable(torch.Tensor(Y_1_JP).float())

    

    

    out2 = model2(inputs2)

    out2 = out2.squeeze(dim=1) # necesario para quitar la dimension intermedia de channel

    loss2 = distance2(out2, targets2)

    loss2.backward()

    optimizer2.step()

    if epoch>(epochs-5):

        print('epoch [{}/{}], loss1:{:.4f} , loss2:{:.4f}  '.format(epoch+1, epochs, loss1.item(),loss2.item()))
#model3 = torch.nn.Sequential(

#    torch.nn.Linear(14, 25),

#    torch.nn.Linear(25, 35),

#    torch.nn.Linear(35, 10),

#    torch.nn.Linear(10, 1)

#)



model3 = torch.nn.Sequential(

    torch.nn.Conv1d(1, 10, kernel_size=8, stride=1),

    torch.nn.Conv1d(10, 6, kernel_size=4, stride=1),

    torch.nn.Conv1d(6, 3, kernel_size=1, stride=1),

    torch.nn.Conv1d(3, 1, kernel_size=2, stride=3)

)



model4 = torch.nn.Sequential(

    torch.nn.Conv1d(1, 10, kernel_size=8, stride=1),

    torch.nn.Conv1d(10, 6, kernel_size=4, stride=1),

    torch.nn.Conv1d(6, 3, kernel_size=1, stride=1),

    torch.nn.Conv1d(3, 1, kernel_size=2, stride=3)

)





distance3 = torch.nn.KLDivLoss()

optimizer3 = torch.optim.Adam(model3.parameters(), lr=1E-4, weight_decay=1E-3)





distance4 = torch.nn.KLDivLoss()

optimizer4 = torch.optim.Adam(model4.parameters(), lr=5E-5, weight_decay=1E-3)

optimizer4.zero_grad()

epochs = 650

for epoch in range(epochs):

    #Training Italy

    X_new3 = np.expand_dims(Italy, 1) 

    inputs3 = torch.autograd.Variable(torch.Tensor(X_new3).float())

    targets3 = torch.autograd.Variable(torch.Tensor(Y_2_IT).float())

    

    optimizer3.zero_grad()

    out3 = model3(inputs3)

    out3 = out3.squeeze(dim=1) # necesario para quitar la dimension intermedia de channel

    loss3 = distance3(out3, targets3)

    loss3.backward()

    optimizer3.step()

    

    #Training Japan

    X_new4 = np.expand_dims(Japan, 1) 

    inputs4 = torch.autograd.Variable(torch.Tensor(X_new4).float())

    targets4 = torch.autograd.Variable(torch.Tensor(Y_2_JP).float())

    

    optimizer4.zero_grad()

    out4 = model4(inputs4)

    out4 = out4.squeeze(dim=1) # necesario para quitar la dimension intermedia de channel

    loss4 = distance4(out4, targets4)

    loss4.backward()

    optimizer4.step()

    if epoch>(epochs-5):

        print('epoch [{}/{}], loss3:{:.4f} , loss4:{:.4f}  '.format(epoch+1, epochs, loss3.item(),loss4.item()))
#NCPM

values1, Y_predicted1 = torch.max(out1.data, 1)

values2, Y_predicted2 = torch.max(out2.data, 1)



plt.figure(figsize=(18, 14))

plt.subplot(2,2,1)

plt.plot(Fechas,values1, label='Model')

plt.plot(Fechas,Y_1_IT, label='Real')

plt.legend()

plt.xlabel('Day')

plt.xticks(Fechas[::30])

plt.ylabel('New Cases per Million')

plt.title('Italian Evolution of the COVID')

          

plt.subplot(2,2,2)

plt.plot(Fechas,values2, label='Model')

plt.plot(Fechas,Y_1_JP, label='Real')

plt.legend()

plt.xlabel('Day')

plt.xticks(Fechas[::30])

plt.ylabel('New Cases per Million')

plt.title('Japanese Evolution of the COVID')



#TDPM

values3, Y_predicted3 = torch.max(out3.data, 1)

values4, Y_predicted4 = torch.max(out4.data, 1)



plt.subplot(2,2,3)

plt.plot(Fechas,values3, label='Model')

plt.plot(Fechas,Y_2_IT, label='Real')

plt.legend()

plt.xlabel('Day')

plt.xticks(Fechas[::30])

plt.ylabel('New Deaths per Million')

plt.title('Italian Evolution of the COVID')

          

plt.subplot(2,2,4)

plt.plot(Fechas,values4, label='Model')

plt.plot(Fechas,Y_2_JP, label='Real')

plt.legend()

plt.xlabel('Day')

plt.xticks(Fechas[::30])

plt.ylabel('New Deathsper Million')

plt.title('Japanese Evolution of the COVID')
#datos

X_new = np.expand_dims(Colombia, 1) 

inputs = torch.autograd.Variable(torch.Tensor(X_new).float())

#Modelo 1 Italia

x_transform_I = model1(inputs)

values_I, Y_predicted_I = torch.max(x_transform_I.data, 1)



#Modelo 2 Japon

x_transform_J = model2(inputs)

values_J, Y_predicted_J = torch.max(x_transform_J.data, 1)

#Modelo 3 Italia

x_transform_I2 = model3(inputs)

values_I2, Y_predicted_I2 = torch.max(x_transform_I2.data, 1)



#Modelo 4 Japon

x_transform_J2 = model4(inputs)

values_J2, Y_predicted_J2 = torch.max(x_transform_J2.data, 1)







plt.figure(figsize=(15, 20))

plt.subplot(2,1,1)

plt.plot(Fechas,values_I, label='Modelo_Italia')

plt.plot(Fechas,values_J, label='Modelo_Japan')

plt.plot(Fechas,Y_1_CO, label='True')

plt.legend()

plt.xlabel('Day')

plt.xticks(Fechas[::14])

plt.ylabel('New Cases per Million')

plt.title('Evolution of the COVID')



plt.subplot(2,1,2)

plt.plot(Fechas,values_I2, label='Modelo_Italia')

plt.plot(Fechas,values_J2, label='Modelo_Japan')

plt.plot(Fechas,Y_2_CO, label='True')

plt.legend()

plt.xlabel('Day')

plt.xticks(Fechas[::14])

plt.ylabel('New Deaths per Million')

plt.title('Evolution of the COVID')