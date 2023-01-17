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
DEBUG = False





#https://www.kaggle.com/theoviel/load-the-totality-of-the-data

dtypes = {

    'MachineIdentifier':                                    'object',

    'ProductName':                                          'object',

    'EngineVersion':                                        'object',

    'AppVersion':                                           'object',

    'AvSigVersion':                                         'object',

    'IsBeta':                                               'int8',

    'RtpStateBitfield':                                     'float16',

    'IsSxsPassiveMode':                                     'int8',

    'DefaultBrowsersIdentifier':                            'float32',  # was 'float16'

    'AVProductStatesIdentifier':                            'float32',

    'AVProductsInstalled':                                  'float16',

    'AVProductsEnabled':                                    'float16',

    'HasTpm':                                               'int8',

    'CountryIdentifier':                                    'int16',

    'CityIdentifier':                                       'float32',

    'OrganizationIdentifier':                               'float16',

    'GeoNameIdentifier':                                    'float16',

    'LocaleEnglishNameIdentifier':                          'int16',  # was 'int8'

    'Platform':                                             'object',

    'Processor':                                            'object',

    'OsVer':                                                'object',

    'OsBuild':                                              'int16',

    'OsSuite':                                              'int16',

    'OsPlatformSubRelease':                                 'object',

    'OsBuildLab':                                           'object',

    'SkuEdition':                                           'object',

    'IsProtected':                                          'float16',

    'AutoSampleOptIn':                                      'int8',

    'PuaMode':                                              'object',

    'SMode':                                                'float16',

    'IeVerIdentifier':                                      'float16',

    'SmartScreen':                                          'object',

    'Firewall':                                             'float16',

    'UacLuaenable':                                         'float64', # was 'float32'

    'Census_MDC2FormFactor':                                'object',

    'Census_DeviceFamily':                                  'object',

    'Census_OEMNameIdentifier':                             'float32', # was 'float16'

    'Census_OEMModelIdentifier':                            'float32',

    'Census_ProcessorCoreCount':                            'float16',

    'Census_ProcessorManufacturerIdentifier':               'float16',

    'Census_ProcessorModelIdentifier':                      'float32', # was 'float16'

    'Census_ProcessorClass':                                'object',

    'Census_PrimaryDiskTotalCapacity':                      'float64', # was 'float32'

    'Census_PrimaryDiskTypeName':                           'object',

    'Census_SystemVolumeTotalCapacity':                     'float64', # was 'float32'

    'Census_HasOpticalDiskDrive':                           'int8',

    'Census_TotalPhysicalRAM':                              'float32',

    'Census_ChassisTypeName':                               'object',

    'Census_InternalPrimaryDiagonalDisplaySizeInInches':    'float32', # was 'float16'

    'Census_InternalPrimaryDisplayResolutionHorizontal':    'float32', # was 'float16'

    'Census_InternalPrimaryDisplayResolutionVertical':      'float32', # was 'float16'

    'Census_PowerPlatformRoleName':                         'object',

    'Census_InternalBatteryType':                           'object',

    'Census_InternalBatteryNumberOfCharges':                'float64', # was 'float32'

    'Census_OSVersion':                                     'object',

    'Census_OSArchitecture':                                'object',

    'Census_OSBranch':                                      'object',

    'Census_OSBuildNumber':                                 'int16',

    'Census_OSBuildRevision':                               'int32',

    'Census_OSEdition':                                     'object',

    'Census_OSSkuName':                                     'object',

    'Census_OSInstallTypeName':                             'object',

    'Census_OSInstallLanguageIdentifier':                   'float16',

    'Census_OSUILocaleIdentifier':                          'int16',

    'Census_OSWUAutoUpdateOptionsName':                     'object',

    'Census_IsPortableOperatingSystem':                     'int8',

    'Census_GenuineStateName':                              'object',

    'Census_ActivationChannel':                             'object',

    'Census_IsFlightingInternal':                           'float16',

    'Census_IsFlightsDisabled':                             'float16',

    'Census_FlightRing':                                    'object',

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

    'HasDetections':                                        'float32',

}



def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage(deep=True).sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage(deep=True).sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df





numerics = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']

numerical_columns = [c for c,v in dtypes.items() if v in numerics]

categorical_columns = [c for c,v in dtypes.items() if v not in numerics]
X_train = pd.read_csv('../input/microsoft-malware-prediction/train.csv', dtype=dtypes)

X_train = reduce_mem_usage(X_train)

X_test = pd.read_csv('../input/microsoft-malware-prediction/test.csv', dtype=dtypes)

X_test = reduce_mem_usage(X_test)



# From timestamps set:

avsig_timestamp = np.load('../input/timestamps/AvSigVersionTimestamps.npy')[()]

osver_timestamp = np.load('../input/timestamps/OSVersionTimestamps.npy')[()]



print('data loaded.')
X_train['DateAvSigVersion'] = X_train['AvSigVersion'].map(avsig_timestamp)

X_train['DateOSVersion'] = X_train['Census_OSVersion'].map(osver_timestamp)

X_train['DateAvSigVersion'] = pd.to_datetime(X_train['DateAvSigVersion'])

X_train['DateOSVersion'] = pd.to_datetime(X_train['DateOSVersion'])



X_test['DateAvSigVersion'] = X_test['AvSigVersion'].map(avsig_timestamp)

X_test['DateOSVersion'] = X_test['Census_OSVersion'].map(osver_timestamp)

X_test['DateAvSigVersion'] = pd.to_datetime(X_test['DateAvSigVersion'])

X_test['DateOSVersion'] = pd.to_datetime(X_test['DateOSVersion'])



print('timestamps mapped.')
X_test_pub = X_test.loc[X_test.DateAvSigVersion <= '2018-10-25', :]

X_test_priv = X_test.loc[X_test.DateAvSigVersion > '2018-10-25', :]



print('public test shape: {}'.format(X_test_pub.shape))

print('private test shape: {}'.format(X_test_priv.shape))

print('fraction of private test split: {:.3f}'.format(X_test_priv.shape[0] / X_test.shape[0]))
dates_intersect = list(set(X_test.DateAvSigVersion.unique()).intersection(X_train.DateAvSigVersion.unique()))

dates_diff = list(set(X_test.DateAvSigVersion.unique()).difference(X_train.DateAvSigVersion.unique()))



dates_os_intersect = list(set(X_test.DateOSVersion.unique()).intersection(X_train.DateOSVersion.unique()))

dates_os_diff = list(set(X_test.DateOSVersion.unique()).difference(X_train.DateOSVersion.unique()))



print('number of dates intersection and difference based on:')

print('AvSig: {}, {}'.format(len(dates_intersect), len(dates_diff)))

print('OSVersion: {}, {}'.format(len(dates_os_intersect), len(dates_os_diff)))
colname = 'DateAvSigVersion'



train_in = X_train.loc[

    X_train[colname].isin(dates_intersect), colname].shape[0] / X_train.shape[0]

train_not = X_train.loc[

    ~X_train[colname].isin(dates_intersect), colname].shape[0] / X_train.shape[0]



test_in = X_test.loc[

    X_test[colname].isin(dates_intersect), colname].shape[0] / X_test.shape[0]

test_not = X_test.loc[

    ~X_test[colname].isin(dates_intersect), colname].shape[0] / X_test.shape[0]



print('train and whole test sets...')

print('based on: {}'.format(colname))

print('fraction of train dates intersection: {:.3f}'.format(train_in))

print('fraction of train dates difference: {:.3f}'.format(train_not))

print('fraction of test dates intersection: {:.3f}'.format(test_in))

print('fraction of test dates difference: {:.3f}'.format(test_not))
colname = 'DateOSVersion'



train_in = X_train.loc[

    X_train[colname].isin(dates_os_intersect), colname].shape[0] / X_train.shape[0]

train_not = X_train.loc[

    ~X_train[colname].isin(dates_os_intersect), colname].shape[0] / X_train.shape[0]



test_in = X_test.loc[

    X_test[colname].isin(dates_os_intersect), colname].shape[0] / X_test.shape[0]

test_not = X_test.loc[

    ~X_test[colname].isin(dates_os_intersect), colname].shape[0] / X_test.shape[0]



print('train and whole test sets...')

print('based on: {}'.format(colname))

print('fraction of train dates intersection: {:.3f}'.format(train_in))

print('fraction of train dates difference: {:.3f}'.format(train_not))

print('fraction of test dates intersection: {:.3f}'.format(test_in))

print('fraction of test dates difference: {:.3f}'.format(test_not))
colname = 'DateAvSigVersion'



test_pub_in = X_test_pub.loc[

    X_test_pub[colname].isin(dates_intersect), colname].shape[0] / X_test_pub.shape[0]

test_pub_not = X_test_pub.loc[

    ~X_test_pub[colname].isin(dates_intersect), colname].shape[0] / X_test_pub.shape[0]



test_priv_in = X_test_priv.loc[

    X_test_priv[colname].isin(dates_intersect), colname].shape[0] / X_test_priv.shape[0]

test_priv_not = X_test_priv.loc[

    ~X_test_priv[colname].isin(dates_intersect), colname].shape[0] / X_test_priv.shape[0]



print('test public and test private..')

print('based on: {}'.format(colname))

print('fraction of public test dates intersection: {:.3f}'.format(test_pub_in))

print('fraction of public test dates difference: {:.3f}'.format(test_pub_not))

print('fraction of private test dates intersection: {:.3f}'.format(test_priv_in))

print('fraction of private test dates difference: {:.3f}'.format(test_priv_not))
train_months = X_train.DateAvSigVersion.apply(lambda x: '{}-{}'.format(x.year, x.month))

test_months = X_test.DateAvSigVersion.apply(lambda x: '{}-{}'.format(x.year, x.month))



df_months = pd.DataFrame(train_months.value_counts()).reset_index()

df_months = df_months.merge(pd.DataFrame(test_months.value_counts()).reset_index(), how='left', on='index')



df_months.sort_values('index')
df_overlap = {}

df_priv_overlap = {}



for c in X_test.columns[1:]:



    train_col_unique = X_train[c].unique()

    test_pub_col_unique = X_test_pub[c].unique()

    test_priv_col_unique = X_test_priv[c].unique()



    col_intersect = set(train_col_unique).intersection(set(test_pub_col_unique))

    col_union = set(train_col_unique).union(set(test_pub_col_unique))

    len_intersect = len(col_intersect)

    len_union = len(col_union)



    df_overlap[c] = len_intersect / len_union

    print('train/pub test, {}, frac intersection: {:.3f}'.format(c, len_intersect / len_union))



    priv_col_intersect = set(train_col_unique).intersection(set(test_priv_col_unique))

    priv_col_union = set(train_col_unique).union(set(test_priv_col_unique))

    priv_len_intersect = len(priv_col_intersect)

    priv_len_union = len(priv_col_union)



    df_priv_overlap[c] = priv_len_intersect / priv_len_union

    print('train/priv test{}, frac intersection: {:.3f}'.format(c, priv_len_intersect / priv_len_union))



df_overlap = pd.DataFrame.from_dict(df_overlap, orient='index').reset_index()

df_overlap.columns = ['colname', 'overlap']

df_overlap.to_csv('./df_cols_pub_overlap.csv', index=False)



df_priv_overlap = pd.DataFrame.from_dict(df_priv_overlap, orient='index').reset_index()

df_priv_overlap.columns = ['colname', 'overlap']

df_priv_overlap.to_csv('./df_cols_priv_overlap.csv', index=False)



df_overlap.head()
df_overlap.loc[df_overlap.overlap < 0.5]
df_priv_overlap.loc[df_priv_overlap.overlap < 0.5]