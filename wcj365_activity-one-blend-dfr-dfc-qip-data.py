import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', 100)  
dtype_dict= {'provfs': str,          # provider CCN numbers should be treated as strings
             'network': str}         # The network numbers should be treated as strings. There are 18 networks.

column_dict ={
    'provfs':'CCN',
    'chainnam':'ChainOwner',
    'network':'Network',
    'provcity': 'City',
    'provname':'Name',
    'state':'StateCode',
    'FIRST_CERTDATE':'InitCertDate',
    'owner_f':'ProfitStatus',
    'sv_Shift_start_after_5_pm':'EveningShift',
    'sv_Practices_Dialyzer_Reuse':'DialyzerReuse',
    'totstas_f':'TotalStations',
    'staffy4_f':'TotalStaff',
    'compl_cond':'ComplianceStatus',
    'allcnty4_f':'TotalPatients', 
    'medicarey4_f':'PctgMedicare',           # % Medicare
    'medpendy4_f':'PctgMedicarePending',     #  % Medicare Pending
    'nonmedy4_f':'PctgNonMedicare',          #  % Non-Medicare
    'agey4_f':'AverageAge',
    'age1y4_f':'PctgAge18',                  # % Age < 18
    'age2y4_f':'PctgAge18t64',               # % Age 18 to 64
    'age3y4_f':'PctgAge64',                  # % Age > 64
    'sexy4_f':'PctgFemale',                  # % Female
    'rac2y4_f':'PctgBlack',                  # % African American
    'eth1y4_f':'PctgHispanic',               # % Hispanic
    'srry4_f':'SRR',                         # Standardized Readmission Ratio
    'CWesarxy4_f': 'PctgESAPrescribed',      # % ESA Prescribed
    'FVhy4_f': 'PctgFluVaccine',             # % patients flu Vaccinated
    'CWavgHGBy4_f': 'AvgHemoglobin',         # Average hemoglobin levels (g/dL), of valid in range adult pt-mths, 2016
    'CWhgb1y4_f': 'PctgHemoglobin10',        # % Hemoglobin < 10 g/dL
    'CWhgb2y4_f': 'PctgHemoglobin10t11',     # % Hemoglobin 10 to 11 g/dL
    'CWhgb3y4_f': 'PctgHemoglobin11t12',     # % Hemoglobin 11 to 12 g/dL
    'CWhgb4y4_f': 'PctgHemoglobin12',        # % Hemoglobin > 12 g/dL
    'CWhdavgufry4_f': 'AvgUFR',              # Average UFR, of valid in range adult HD pt-mths, 2016'
    'CWhdufr1y4_f': 'PctgUFRLE13',           # % of adult HD pt-mths with UFR <= 13, 2016'
    'CWhdufr2y4_f': 'PctgUFRGT13',           # % of adult HD pt-mths with UFR > 13, 2016'
    'CWhdavgktvy4_f': 'AvgKtV',              # Average Kt/V, of valid in range adult HD pt-mths, 2016
    'CWhdktv1y4_f':  'PctgKtV12',            # % Kt/V < 1.2
    'CWhdktv2y4_f':  'PctgKtV12t18',         # % Kt/V 1.2 to 1.8
    'CWhdktv3y4_f': 'PctgKtV18',             # % Kt/V >= 1.8
    'CWhdktv4y4_f': 'PctgKtVOther',          # % Kt/V Missing or Out of Range
    'CWavgPy4_f':'AvgSerumPhosphorous',      # Avg serum phosphorous (mg/dL), of valid in range adult pt-mths, 2016
    'CWP5y4_f': 'PctgSerumPhosphorous70',        # % of adult pt-mths with serum phosphorous > 7.0 mg/dL, 2016
    'CWavgUnCay4_f': 'AvgCalciumUncorrected',    # Avg uncorrected calcium (mg/dL), of valid in range adult pt-mths, 2016
    'CWunCa3y4_f': 'PctgCalciumUncorrected102',  # % of adult pt-mths with uncorrected calcium > 10.2 mg/dL, 2016
    'ppavfy4_f': 'PctgFistula',              # % of patients receiving treatment w/ fistulae, 2016
    'ppcg90y4_f': 'PctgCatheterOnly90',      # % of patients with catheter only >= 90 days, 2016
}

dfr=pd.read_csv("../input/DFR-FY-2018.csv", parse_dates=True, dtype=dtype_dict, usecols=column_dict.keys())
dfr.rename(columns=column_dict, inplace=True)
print("\nThe DFR data frame has {0} rows or facilities and {1} variables or columns are selected out of over 3000.\n".format(dfr.shape[0], dfr.shape[1]))
dfr.head()   # take a look at the first five facilities
dtype_dict_dfc= {'Provider Number': str,       # Provider Number and Zip need to be treated as string  
                 'Zip': str}                   # or leading zeros will be dropped during the import.
column_dict_dfc={'Provider Number': 'CCN', 
                 'Zip': 'ZipCode'}             # We only need CCN and zip code column
dfc=pd.read_csv("../input/DFC-CY2018.csv", dtype=dtype_dict_dfc, usecols=column_dict_dfc.keys() )
dfc.rename(columns=column_dict_dfc, inplace=True)
print("\nThe DFC dataset has {0} rows or facilities and {1} variables or columns are selected.\n".format(dfc.shape[0], dfc.shape[1]))
dfc.info()
dfc.sample(5)    # display a random sample of five observations or facilities
dfr_ccn = set(dfr['CCN'])
dfc_ccn = set(dfc["CCN"])
print("There are {0} facilities in DFC that are not in DFR".format(len(dfc_ccn - dfr_ccn)))
dfr = pd.merge(dfr, dfc, on='CCN', how='left') 
dfr.shape
print("There are {0} facilities in the DFR dataframe without a zip code".format(dfr[dfr["ZipCode"].isnull()].shape[0]))
dtype_dict_qip= {'CMS Certification Number (CCN)': str,
                 'Zip Code': str}
column_dict_qip={'CMS Certification Number (CCN)': 'CCN', # We only need the CCN and Zip Code columns
                 'Zip Code': 'ZipCode'}
qip=pd.read_csv("../input/ESRD-QIP-PY2018.csv", dtype=dtype_dict_qip, usecols=column_dict_qip.keys())
qip.rename(columns=column_dict_qip, inplace=True)
print("\nThe QIP dataset has {0} rows or facilities and {1} variables or columns are selected.\n".format(qip.shape[0], qip.shape[1]))
qip.info()
qip.tail(5)  # take a look at the last five facilities
dfr_ccn = set(dfr['CCN'])
qip_ccn = set(qip["CCN"])
print("There are {0} facilities in QIP dataframe that are not in DFR data frame".format(len(qip_ccn - dfr_ccn)))
dfr = pd.merge(dfr, qip, on="CCN", how="left")  # merge QIP dataset with DFR dataset to get zip code from QIP dataset
dfr.shape
dfr.columns    # display all columns in the DFR dataframe
print("The new dataset has {0} facilities without zip code based on ZipCode_x column.".format(dfr[dfr['ZipCode_x'].isnull()].shape[0]))
dfr['ZipCode_x'].fillna(dfr['ZipCode_y'], inplace=True)
print("The new dataset has {0} facilities without zip code".format(dfr[dfr['ZipCode_x'].isnull()].shape[0]))
dfr.drop("ZipCode_y", axis=1, inplace=True)
dfr.rename(columns={'ZipCode_x':'ZipCode'}, inplace=True)
dfr.shape
dfr.sample(5)  # display a random sample of five facilities
dfr.to_csv("InterimDataset.csv")
