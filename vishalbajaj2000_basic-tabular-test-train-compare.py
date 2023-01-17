import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import gc

import os

print(os.listdir("../input"))

pd.set_option('max_columns', 150)

pd.set_option('max_rows', 150)

pd.set_option('max_colwidth', 400)

pd.set_option('max_seq_items', 400)

pd.set_option('max_info_rows', 150)

pd.set_option('max_info_columns', 150)
#Borrowed from Bojan's kernel

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
def df_reader(path, chunksize,usecols=[i for i in dtypes]):

    reader = train_reader = pd.read_csv(path, chunksize=chunksize, dtype=dtypes, usecols = usecols)

    dflist = []

    for df_part in reader:

        dflist.append(df_part)

    data = pd.concat(dflist,sort=False)

    return data
%%time

train = df_reader('../input/train.csv', chunksize=2000000)

print(train.shape, '--- Size is ---', train.size)
%%time

test = df_reader('../input/test.csv', chunksize=2000000, usecols = [i for i in dtypes][:-1])

print(train.shape, '--- Size is ---', train.size)
gc.collect()
def train_test_compare(train=None, test=None, maxValues=30, maxValComp=2000, writefile=True, filename='Train_Test_DetailedCompare20190207', debug=False):

    """

    Given one/(two datasets compare the data sets) and return pertenant features for basic EDA.

      ```

      Args:

        train: required, "train" pandas dataframe

        test: optional, default = None, "test" pandas dataframe

        maxValues: required, Default 30, the max unique values to display in the comparison column.

        maxValComp: optional, default 2000, number of unique values to compare in a column between 2 datasets. 

                    Setting this too hig will make the process longer!

        writefile: Boolean, default = True

        filename: optional, default = 'Train_Test_DetailedCompare', the file name without the file extention.

        debug: optional, prints out the column name.



      Returns:

        Pandas dataframe with test and train set compared, or just the dataset illustrated.



    """

    if train is None:

        print('No train set provided')

        return None



    if test is None:

        objcol = train.columns[1:]

    else:

        objcol = list(set(train.columns[1:].tolist() + test.columns[1:].tolist()))



    collist=[]



    for col in objcol:

        if debug == True:

            print(col)

        only_train = {}

        only_test = {}

        tempx = {}

        tempy = {}

        

        TR_ColSize = round(train[col].memory_usage() / 1024**2, 2)

        train_ucnt = train[col].nunique()        

        train_na = train[col].isna().sum()        

        trainNanP = round(train_na * 100 / train.shape[0],3)

        if train[col].nunique() > maxValues:

            train_u = 'Too many values!'

        else:

            train_u = set(train[col].unique())



        if test is None or col not in test.columns[1:].tolist():

            TS_ColSize = 'N/A'

            test_ucnt = 'N/A'

            test_na = 'N/A'

            testNanP = 'N/A'

            test_u = 'N/A'

            only_test = {'N/A'}

            only_train = {'N/A'}

        else:

            TS_ColSize = round(test[col].memory_usage() / 1024**2, 2)

            test_ucnt = test[col].nunique() 

            test_na = test[col].isna().sum()

            testNanP = round(test_na * 100 /  test.shape[0],3)

            

            if test[col].nunique() > maxValues:

                test_u = 'Too many values!'

            else:

                test_u = set(test[col].unique())



            if train_ucnt > maxValComp or test_ucnt > maxValComp:

                only_train = ['Too many values to compare']

                only_test = ['Too many values to compare']



            else:

                train_ux = set(train[col].unique())

                test_ux = set(test[col].unique())

                train_diff = train_ux - test_ux

                test_diff = test_ux - train_ux

            

                if train_diff == set():

                    only_train = ['No diff']

                elif len(train_diff) > maxValues:

                    only_train = ['Too many values']

                elif len(train_diff) < maxValues: 

                    for val in train_diff:

                        if pd.notna(val) == False:

                            continue

                        else:

                            tempx[val] = (train[col].values == val).sum()

                    val_sumx = train[col][train[col].isin(list(train_diff))].count()

                    only_train = [('TOTALDIFF', val_sumx )]

                    only_train.append(sorted(tempx.items(), key=lambda kv: kv[1], reverse=True))

                    

                if test_diff == set():

                    only_test = ['No diff']

                elif len(test_diff) > maxValues:

                    only_test = ['Too many values']

                elif len(test_diff) < maxValues: 

                    for val in test_diff:

                        if pd.notna(val) == False:

                            continue

                        else:

                            tempy[val] = (test[col].values == val).sum()

                    val_sumy = test[col][test[col].isin(list(test_diff))].count()

                    only_test = [('TOTALDIFF', val_sumy )]

                    only_test.append(sorted(tempy.items(), key=lambda kv: kv[1], reverse=True))



        collist.append([col,train[col].dtypes,TR_ColSize,train_na,test_na,trainNanP, testNanP, train_ucnt, test_ucnt, train_u, test_u, only_train, only_test])

        

    comparedf = pd.DataFrame(collist, columns=['Column','Col_Type','TR_ColSizeMB', 'TR_NaN','TS_NaN','TR_NaN%','TS_NaN%','TR_UnqCount','TS_UnqCount','TR_UnqValues','TS_UnqValues','Train_Only', 'Test_Only'])     

        

    if writefile == True:

        comparedf.to_excel(str(filename) + '.xlsx')



    return comparedf
%%time

Train_test_comp = train_test_compare(train, test, maxValues=150, writefile=True, debug=True)
Train_test_comp
print(os.listdir("../working"))