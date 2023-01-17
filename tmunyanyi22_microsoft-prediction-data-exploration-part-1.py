###Loading in useful packages



#for linear algebra

import numpy as np



#for data manipulation

import pandas as pd



#for plotting

import matplotlib.pyplot as pp

import seaborn as sns

%matplotlib inline



#For surpressing warnings

import warnings

warnings.filterwarnings('ignore')



#For opening Zip files

import zipfile as zf
dtypes = {

        'MachineIdentifier':                                    'category',

        'ProductName':                                          'category',

        'EngineVersion':                                        'category',

        'AppVersion':                                           'category',

        'AvSigVersion':                                         'category',

        'IsBeta':                                               'int8',

        'RtpStateBitfield':                                     'float16',

        'IsSxsPassiveMode':                                     'int8',

        'DefaultBrowsersIdentifier':                            'float16',

        'AVProductStatesIdentifier':                            'float32',

        'AVProductsInstalled':                                  'float16',

        'AVProductsEnabled':                                    'float16',

        'HasTpm':                                               'int8',

        'CountryIdentifier':                                    'int16',

        'CityIdentifier':                                       'float32',

        'OrganizationIdentifier':                               'float16',

        'GeoNameIdentifier':                                    'float16',

        'LocaleEnglishNameIdentifier':                          'int8',

        'Platform':                                             'category',

        'Processor':                                            'category',

        'OsVer':                                                'category',

        'OsBuild':                                              'int16',

        'OsSuite':                                              'int16',

        'OsPlatformSubRelease':                                 'category',

        'OsBuildLab':                                           'category',

        'SkuEdition':                                           'category',

        'IsProtected':                                          'float16',

        'AutoSampleOptIn':                                      'int8',

        'PuaMode':                                              'category',

        'SMode':                                                'float16',

        'IeVerIdentifier':                                      'float16',

        'SmartScreen':                                          'category',

        'Firewall':                                             'float16',

        'UacLuaenable':                                         'float32',

        'Census_MDC2FormFactor':                                'category',

        'Census_DeviceFamily':                                  'category',

        'Census_OEMNameIdentifier':                             'float16',

        'Census_OEMModelIdentifier':                            'float32',

        'Census_ProcessorCoreCount':                            'float16',

        'Census_ProcessorManufacturerIdentifier':               'float16',

        'Census_ProcessorModelIdentifier':                      'float16',

        'Census_ProcessorClass':                                'category',

        'Census_PrimaryDiskTotalCapacity':                      'float32',

        'Census_PrimaryDiskTypeName':                           'category',

        'Census_SystemVolumeTotalCapacity':                     'float32',

        'Census_HasOpticalDiskDrive':                           'int8',

        'Census_TotalPhysicalRAM':                              'float32',

        'Census_ChassisTypeName':                               'category',

        'Census_InternalPrimaryDiagonalDisplaySizeInInches':    'float16',

        'Census_InternalPrimaryDisplayResolutionHorizontal':    'float16',

        'Census_InternalPrimaryDisplayResolutionVertical':      'float16',

        'Census_PowerPlatformRoleName':                         'category',

        'Census_InternalBatteryType':                           'category',

        'Census_InternalBatteryNumberOfCharges':                'float32',

        'Census_OSVersion':                                     'category',

        'Census_OSArchitecture':                                'category',

        'Census_OSBranch':                                      'category',

        'Census_OSBuildNumber':                                 'int16',

        'Census_OSBuildRevision':                               'int32',

        'Census_OSEdition':                                     'category',

        'Census_OSSkuName':                                     'category',

        'Census_OSInstallTypeName':                             'category',

        'Census_OSInstallLanguageIdentifier':                   'float16',

        'Census_OSUILocaleIdentifier':                          'int16',

        'Census_OSWUAutoUpdateOptionsName':                     'category',

        'Census_IsPortableOperatingSystem':                     'int8',

        'Census_GenuineStateName':                              'category',

        'Census_ActivationChannel':                             'category',

        'Census_IsFlightingInternal':                           'float16',

        'Census_IsFlightsDisabled':                             'float16',

        'Census_FlightRing':                                    'category',

        'Census_ThresholdOptIn':                                'float16',

        'Census_FirmwareManufacturerIdentifier':                'float16',

        'Census_FirmwareVersionIdentifier':                     'float32',

        'Census_IsSecureBootEnabled':                           'int8',

        'Census_IsWIMBootEnabled':                              'float16',

        'Census_IsVirtualDevice':                               'float16',

        'Census_IsTouchEnabled':                                'int8',

        'Census_IsPenCapable':                                  'int8',

        'Census_IsAlwaysOnAlwaysConnectedCapable':              'float16',

        'Wdft_IsGamer':                                         'float16',

        'Wdft_RegionIdentifier':                                'float16',

        'HasDetections':                                        'int8'

        }

#Change the training and testing sets to pickle format



#I have commented these out because I have already completed this step and these files are avaliable on my drive now.



#pd.read_csv(zf.ZipFile("/content/drive/My Drive/Microsoft/train.zip").open("train.csv"), dtype=dtypes).to_pickle("/content/drive/My Drive/Microsoft/train.pkl")

#pd.read_csv(zf.ZipFile("/content/drive/My Drive/Microsoft/test.zip").open("test.csv"), dtype=dtypes).to_pickle("/content/drive/My Drive/Microsoft/test.pkl")
#Opening the saved pickles - which contain the data :)



train = pd.read_pickle("../input/mctrain/train.pkl")

test =pd.read_pickle("../input/mctest/test.pkl")
#Looking at shape of training data, i.e number of rows and columns



print("Training data dimensions",train.shape)

print("Testing data dimension",test.shape)
#Training variable data types

train.info()
#Make a dictionary of all the variables

all_the_vars = list(train.drop(['MachineIdentifier','HasDetections'],axis=1))



#Make a list of categorical variables

cat_vars = [i for i in all_the_vars if (train[i].dtype.name == 'category') | (train[i].dtype.name == 'object')]



#Make a list of binary variables

bin_vars = [i for i in all_the_vars if len(train[i].value_counts()) <= 2 ]



#Make a list of pure numerical variables

num_vars = [i for i in all_the_vars if (i not in cat_vars) & (i not in bin_vars)]



print('We have ',len(all_the_vars),' explanatory variables')

print('We have ',len(cat_vars),' categorical variables')

print('We have ',len(bin_vars),' binary variables')

print('We have ',len(num_vars),' numerical variables')

#Let's examine the distribution of HasDetections

j = train['HasDetections'].value_counts()

j = j/len(train)

j.plot.bar()

pp.title('Distribution of HasDetections')

pp.xlabel('HasDetections values')

pp.ylabel('Proportion')

pp.show()
for i in cat_vars:

  length = len(train)

  j = train[i].value_counts() / length

  j = j.sort_values(ascending=False)

  j = j.iloc[:10]

  x = list(j.index)

  y = list(j.values)

  z= list()



  for j in x:

    z.append(train['HasDetections'].loc[train[i]==j].mean())

  



  fig, ax1 = pp.subplots()

  ax1.bar(x,y)

  pp.xticks(x,y)

  locs, labels = pp.xticks()

  pp.setp(labels, rotation=90)

  pp.title(i)

  pp.ylabel('Proportion')

  pp.xlabel(i + ' values')



  ax2 = ax1.twinx()

  ax2.plot(x,z,'r',linestyle='-', marker='o')

  ax1.grid(False)

  ax2.grid(False)

  pp.ylabel('P(HasDetections == 1)')



  pp.show()

  

for i in bin_vars:

  length = len(train)

  j = train[i].value_counts() / length

  j = j.sort_values(ascending=False)

  x = list(j.index)

  y = list(j.values)

  z= list()



  for j in x:

    z.append(train['HasDetections'].loc[train[i]==j].mean())

  



  fig, ax1 = pp.subplots()

  ax1.bar(x,y)

  pp.title(i)

  pp.ylabel('Proportion')

  pp.xlabel(i + ' values')



  ax2 = ax1.twinx()

  ax2.plot(x,z,'r',linestyle='-', marker='o')

  ax1.grid(False)

  ax2.grid(False)

  pp.ylabel('P(HasDetections == 1)')



  pp.show()
import gc

gc.collect()
for i in num_vars:

  length = len(train)

  j = train[i].value_counts() / length

  j = j.sort_values(ascending=False)

  x = np.array(j.index)

  y = np.array(j.values)

  z= list()



  ax = sns.kdeplot(train[i].loc[train['HasDetections']==1],label='HasDetections ==1',color='r')

  ax = sns.kdeplot(train[i].loc[train['HasDetections']==0],label='HasDetections ==0',color='b')

  pp.title(i)

  pp.ylabel('Density')

  pp.xlabel(i+' values')



  

  pp.show()
#Let's creating a series which contains the proportion of missing values for each variable

mis_val = 100 * train.isnull().sum()/len(train)



#Let's view variables with missing values

mis_val[mis_val >0].sort_values(ascending=False)

#Save them to separate pickles

#train.to_pickle("/content/drive/My Drive/Microsoft/train.pkl")

#test.to_pickle("/content/drive/My Drive/Microsoft/test.pkl")