import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', 50)  
dtype_dict= {'CCN': str,    
             'Network': str, 
             'ZipCode': str}

dfr=pd.read_csv("../input/activity-one-blend-dfr-dfc-qip-data/InterimDataset.csv", parse_dates=True, dtype=dtype_dict)
print("\nThe DFR data frame has {0} rows or facilities and {1} variables or columns\n".format(dfr.shape[0], dfr.shape[1]))
dfr.head()
dfr.drop(columns='Unnamed: 0', axis=1, inplace=True)  # remove this unnamed column
dfr.head()
dtype_dict_acs= {'GEO.id2': str}
column_dict_acs={'GEO.id2': 'ZipCode',
                 'HC03_VC50': 'PctgBlackACS',
                 'HC03_VC88': 'PctgHispanicACS'}
acs=pd.read_csv("../input/rmudsc/ACS_16_5YR_DP05_with_ann.csv", skiprows=[1], dtype=dtype_dict_acs, usecols=column_dict_acs.keys() )
acs.rename(columns=column_dict_acs, inplace=True)
print("\nThe DP05 ACS dataset has {0} rows or zip codes and {1} variables or columns are selected.\n".format(acs.shape[0], acs.shape[1]))
acs.info()
acs.sample(5)
dfr = pd.merge(dfr, acs, on='ZipCode', how='left') 
dfr.shape
# First, we import the DFC dataset. We only need CCN and zip code column.

dtype_dict_acs= {'GEO.id2': str}
column_dict_acs={'GEO.id2': 'ZipCode',
                 'HC03_VC07': 'UnemploymentRate',
                 'HC03_VC161': 'PctgFamilyBelowFPL'}
acs=pd.read_csv("../input/rmudsc/ACS_16_5YR_DP03_with_ann.csv", skiprows=[1], dtype=dtype_dict_acs, usecols=column_dict_acs.keys() )
acs.rename(columns=column_dict_acs, inplace=True)
print("\nThe DP03 ACS dataset has {0} rows or zip codes and {1} variables or columns are selected.\n".format(acs.shape[0], acs.shape[1]))
acs.info()
acs.sample(5)
dfr = pd.merge(dfr, acs, on='ZipCode', how='left') 
dfr.shape
dtype_dict_acs= {'GEO.id2': str}
column_dict_acs={'GEO.id2': 'ZipCode',
                 'HC03_VC173': 'PctgPoorEnglish'
                }
acs=pd.read_csv("../input/rmudsc/ACS_16_5YR_DP02_with_ann.csv", skiprows=[1], dtype=dtype_dict_acs, usecols=column_dict_acs.keys() )
acs.rename(columns=column_dict_acs, inplace=True)
print("\nThe DP02 ACS dataset has {0} rows or zip codes and {1} variables or columns are selected.\n".format(acs.shape[0], acs.shape[1]))
acs.info()
acs.sample(5)
dfr = pd.merge(dfr, acs, on='ZipCode', how='left') 
dfr.shape
dfr.to_csv("InterimDataset2.csv")
