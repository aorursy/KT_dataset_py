import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', 60)
dtype_dict= {'CCN': str,    
             'Network': str, 
             'ZipCode': str}

dfr=pd.read_csv("../input/activity-two-blend-acs-data/InterimDataset2.csv", parse_dates=True, dtype=dtype_dict)
print("\nThe DFR data frame has {0} rows or facilities and {1} variables or columns\n".format(dfr.shape[0], dfr.shape[1]))
dfr["ChainOwner"].describe()
len(dfr[dfr['ChainOwner'].isnull()])  
chain_dict = {'DAVITA': 'DaVita',
              'FRESENIUS MEDICAL CARE': 'FMC'}      

def get_owner(owner):
    if owner is np.NaN:
        return 'Independent'
    elif owner in chain_dict.keys():
        return chain_dict[owner]
    else:
        return "Other Chains"

dfr["ChainOwner"] = dfr["ChainOwner"].apply(get_owner)      
dfr['ChainOwner'].value_counts()
dfr['ChainOwner'].value_counts().sort_index() # sort by index. This is the default. so we can omit .sort_index()
dfr['ChainOwner'].value_counts().sort_values() # sort by value
plt.figure(figsize=(10,6))
plt.title("Numbere of Facilities by Chain Owner")
dfr['ChainOwner'].value_counts().plot(kind='bar')   # This is Pandas plot, not as fancy as Seaborn
plt.figure(figsize=(10,6))
# plt.ylabel("Number of Facilities")     # did not work
plt.xticks(rotation=-45)                                 # control the display of the categories
sort_order = dfr['ChainOwner'].value_counts().index
ax = sns.countplot(x='ChainOwner', data=dfr, order=sort_order)   # This is Seaborn plot
ax.set(ylabel='Number of Facilities', xlabel="Dialysis Chain Owner")
dict2 = {
(1,879): 'Short Term Hospitals',
(2000,2299): 'Long Term Hospitals',
(2300,2499): 'Hospital-Based Facilities',
(2500,2899): 'Non-Hospital Facilities',
(2900,2999): 'Non-Hospital Facilities (Special Purpose)',
(3300,3399): 'Children’s Hospitals',
(3500,3699): 'Hospital Satellites Facilities',
(3700,3799): 'Hospital-Based Facilities (Special Purpose)'
       }

dict = {
(1,879): 'Hospitals',
(2000,2299): 'Hospitals',
(2300,2499): 'Hospital-Based Facilities',
(2500,2899): 'Non-Hospital Facilities',
(2900,2999): 'Non-Hospital Facilities',
(3300,3399): 'Hospitals',
(3500,3699): 'Hospital Satellites Facilities',
(3700,3799): 'Hospital-Based Facilities'
       }

def get_facility_type(ccn):
    last_four = int(ccn[2:6])
    fac_type = 'Unknown'
    for key in dict: 
        lower_bound, upper_bound = key # unpacking the lower and upper bounds in the the range
        if lower_bound <= last_four and last_four <= upper_bound:
            fac_type = dict[key]
            break
        else: 
            next
    return fac_type

dfr["HospitalAffiliation"] = dfr['CCN'].apply(get_facility_type)    # how elegant
dfr["HospitalAffiliation"].value_counts()
dfr[dfr['HospitalAffiliation'].isnull()]
plt.figure(figsize=(10,6))
# plt.ylabel("Number of Facilities")     # did not work
plt.xticks(rotation=-45)                                 # control the display of the categories
sort_order = dfr['HospitalAffiliation'].value_counts().index
ax = sns.countplot(x='HospitalAffiliation', data=dfr, order=sort_order)   # This is Seaborn plot
ax.set(ylabel='Number of Facilities', xlabel="Hospital Affiliation")
dfr["StateCode"].describe()
dfr['StateCode'].unique()
column_dict ={'State Code':'StateCode'}  # change the column name to match with DFR dataset
region = pd.read_csv("../input/rmudsc/Census-Bureau-Regions.csv")
region.rename(columns=column_dict, inplace=True)
region.info()
region.head()
region.Region.value_counts()
plt.figure(figsize=(10,6))
region.Region.value_counts().plot(kind="bar")
set(dfr['StateCode'].unique()) - set(region['StateCode'].unique())
dfr = pd.merge(dfr, region, on='StateCode', how='left')
dfr.Region.unique()
dfr[dfr["Region"].isnull()].sample(3)
dfr['Region'].fillna('Oversea', inplace=True)
dfr.Region.unique()
dfr['Division'].unique()
# replace the null value with "Oversea" to represent the oversea territories
dfr['Division'].fillna('Oversea', inplace=True)
dfr.Division.unique()
dtype_dict={'ZIPCODEN':str}
column_dict={'ZIPCODEN': 'ZipCode',
            'RUCA30': 'RUCA'}
rural=pd.read_csv("../input/rmudsc/URCA-310.csv", dtype=dtype_dict, usecols=column_dict.keys())
rural.rename(columns=column_dict, inplace=True)
rural.info()
dfr = pd.merge(dfr, rural, on='ZipCode', how='left')
dfr.shape
def get_urban(ruca):
    if ruca < 4:      # 1.x, 2.x, and 3.x
        return 'Metropolitan'
    elif ruca < 7:    # 4.x, 5.x, and 6.x
        return 'Micropolitan'   
    elif ruca < 10:   # 7.x, 8.x, and 9.x
        return "Small Town"
    elif ruca < 11:   # 10.x
        return "Rural Area"
    else:             # NaN
        return 'Unknown'
# 1,2,and 3 are considered urban, 4 to 10 are considered rural.    

dfr['Urbanicity'] = dfr['RUCA'].apply(get_urban)
#dfr[dfr['Urbanicity'] == 'Unknown']
dfr['Urbanicity'].unique()
dfr.Urbanicity.value_counts().plot(kind="bar")
def get_urban(ruca):
    if ruca < 4:      # 1.x, 2.x, and 3.x
        return 'Urban'
    elif ruca < 11:   # 4 to 10
        return "Rural"
    else:             # NaN
        return 'Unknown'
# 1,2,and 3 are considered urban, 4 to 10 are considered rural.    

dfr['Urbanicity2'] = dfr['RUCA'].apply(get_urban)
dfr.Urbanicity2.value_counts().plot(kind="bar")
dfr['InitCertDate2'] =  pd.to_datetime(dfr['InitCertDate'], format='%d%b%Y')
dfr[['InitCertDate', 'InitCertDate2']].sample(5)
dec_31_2016 = pd.to_datetime('2016-12-31')
dfr['YearsInOps'] =  ((dec_31_2016 - dfr['InitCertDate2'])/np.timedelta64(1, 'Y'))
#dfr['Months In Ops'] =  ((dec_31_2016 - dfr['Init. Cert. Date 2'])/np.timedelta64(1, 'M'))
dfr[dfr['YearsInOps'].isnull() == True]
#  Rreview all columns 
dfr.columns
# Drop the unused ones
drop_columns = ["Unnamed: 0", 
                "RUCA",
                'InitCertDate',
                'InitCertDate2']
                
dfr.drop(drop_columns, axis=1, inplace=True)  
dfr.columns
# Review data elements from five random facilities 
dfr.sample(5)
# DialyzerReuse has missing values
# Python built-in number NaN is Not-a-Numberrepresenting missing values)
dfr['DialyzerReuse'].dtype
dfr['DialyzerReuse'].describe()
# Find out how many facilities do not reuse dialyzer
len(dfr[dfr['DialyzerReuse'].isnull()])  
# Convert the data from float to string
dfr['DialyzerReuse'] = dfr['DialyzerReuse'].astype(str)
dfr['DialyzerReuse'].describe()
def convert_dialyzer_reuse(dialyzer_reuse):
    if dialyzer_reuse == '1.0':    
        return 'Dialyzer Reuse'
    else:             # NaN
        return 'No Dialyzer Reuse'
dfr['DialyzerReuse'] = dfr['DialyzerReuse'].apply(convert_dialyzer_reuse)
dfr['DialyzerReuse'].describe()
## EveningShift has missing values
# Python built-in number NaN is Not-a-Numberrepresenting missing values)
dfr['EveningShift'].describe()
# Find out how many facilities do not offer evening shift
len(dfr[dfr['EveningShift'].isnull()])  
# Convert the data from float to string
dfr['EveningShift'] = dfr['EveningShift'].astype(str)
dfr['EveningShift'].describe()
def convert_evening_shift(evening_shift):
    if evening_shift == '1.0':    
        return 'Evening Shift'
    else:             # NaN
        return 'No Evening Shift'
dfr['EveningShift'] = dfr['EveningShift'].apply(convert_evening_shift)
dfr['EveningShift'].describe()
dfr['ComplianceStatus'].dtype
dfr['ComplianceStatus'] = dfr['ComplianceStatus'].astype(str)
dfr['ComplianceStatus'].describe()
dfr['ComplianceStatus'].value_counts()
def convert_compliance(status):
    if status == '0.0':     
        return 'Compliant'
    elif status == '1.0':   
        return "Not Compliant"
    elif status == '2.0':  
        return "Not Compliant"
    else:                          # NaN
        return 'Unknown'
dfr['ComplianceStatus'] = dfr['ComplianceStatus'].apply(convert_compliance)
dfr.to_csv("DialysisCareQualityData.csv")