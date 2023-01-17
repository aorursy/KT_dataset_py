%matplotlib inline
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 
import numpy as np 
import os 
import pandas as pd 
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
import re
from dateutil.parser import parse

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
for dirname, _, filenames in os.walk('../data/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# get list of countries:
base_file = "../data/worldometer/worldometer-confirmed-cases-and-deaths-by-country-territory-or-conveyance.csv"
df_base = pd.read_csv(base_file)
Countries_list = df_base.country.unique()

print ('List of available countries: \n', Countries_list)

def date_replacer(df):
    Col_to_str = []
    Col_to_date = []
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_datetime(df[col])
                Col_to_date.append(col)
            except ValueError:
                Col_to_str.append(col)
                pass

    Col_to_replacement = Col_to_date 
    
    for item_ in Col_to_replacement:
        
        try:
            df['year'] = pd.to_datetime(df[item_]).dt.year
            df['month'] = pd.to_datetime(df[item_]).dt.month
            df['day'] = pd.to_datetime(df[item_]).dt.day
            df['WeekOfYear'] = pd.to_datetime(df[item_]).dt.weekofyear
            df['DayOfWeek'] = pd.to_datetime(df[item_]).dt.dayofweek
        except:
            pass

    df.drop(Col_to_replacement, axis=1, inplace=True)
    
    return df

#
def get_rows(df, requested_str, theKey):
    
    my_ = pd.DataFrame()
    for col in df.columns:
        if is_string_dtype(df[col]):
            try:
                my_ = df[df[col]==requested_str].copy()
                if my_.shape[0] > 0:
                    my_.rename(columns={col: theKey}, inplace=True)
                    break
            except:
                pass
                
    return my_


MyCountry = 'Spain' # requested country

Key='Country'
dir_prefix = '../data/'
Merge_list = [Key, 'year','month','day']

year = [2019 for x in range(16+31)]
day = [16+x for x in range(16)] # december
month = [12 for x in range(len(day))] # december
year = [2019 for x in range(len(day))] # 2019
day.extend([x for x in range(31) ]) # January
month.extend([1 for x in range(31)]) # January
year.extend([2020 for x in range(31)]) # 2020
day.extend([x for x in range(29) ]) # February
month.extend([2 for x in range(29)]) # February
year.extend([2020 for x in range(29)]) # 2020
day.extend([x for x in range(31) ]) # March
month.extend([3 for x in range(31)]) # March
year.extend([2020 for x in range(31)]) # 2020
Country_ = [MyCountry for x in range(len(year))]

mySelection_ = pd.DataFrame(columns=Merge_list)
mySelection_[Key] = Country_
mySelection_['year'] = year
mySelection_['month'] = month
mySelection_['day'] =  day
myDirs = ['ECDC','HDE','OpenTable','county_health_rankings','covid_tracking_project',
          'github','johns_hopkins','un_world_food_programme','us_cdc','worldometer',
          'our_world_in_data','ihme','harvard_global_health_institute','esri_covid-19','coders_against_covid']
# check shapes:
mySelection_ = pd.DataFrame() 

ind_ = 0
for dirname_ in myDirs:
  dir_ = dir_prefix + dirname_
  print ('start: dir = ', dir_)
  for dirname, _, filenames in os.walk(dir_):
      for filename in filenames:
        myFile = os.path.join(dirname, filename)
        print('file  = ',myFile)
        filename_ = os.path.basename(myFile)
        df = pd.read_csv(myFile)
        mytmp_ = get_rows(df, MyCountry, Key).copy()
        # remove cols with all nans
        mytmp_.dropna(axis=1, how='all',inplace=True)
        
        list_target = ['year','month','day']
        list_columns_mytmp_ = mytmp_.columns.tolist()
        check_columns_mytmp_ = all(elem in list_columns_mytmp_ for elem in list_target)
        res = [i for i in list_columns_mytmp_ if 'year' in i] 
        if check_columns_mytmp_==False:
            # delete those columns:
            mytmp_.drop(res, axis=1, inplace=True)
        
        if mytmp_.shape[0] > 0:
            # replace date to Year, Month and Day:
            mytmp_ = date_replacer(mytmp_)
            # remove duplicated rows:
            mytmp_.drop_duplicates(keep='last', inplace=True)
            
            print ('mytmp_.shape = ',mytmp_.shape)
            
            groups = mytmp_.groupby(np.arange(len(mytmp_.index)))
            for (frameno, frame) in groups:
                # remove columns with nan:
                frame.dropna(axis=1, how='all', inplace=True)
                
                if ind_ == 0:
                    mySelection_ = frame.copy()
                
                else:
                    try:
                        mySelection_ = pd.merge_ordered(mySelection_, frame, left_by=Key, how='outer')
                    except:
                        print ('PASS')
                        pass
                
                ind_ += 1

        # drop duplicated rows:
        mySelection_.drop_duplicates(keep='last', inplace=True)
        # drop columns with nans:
        mySelection_.dropna(axis=1, how='all', inplace=True)
        print('mytmp_, mySelection_: shapes = ',mytmp_.shape, mySelection_.shape)   
            
        
        
mySelection_.dropna(subset=['year','month','day','cases','deaths'], how='all', inplace=True)
mySelection_.dropna(subset=['cases'], how='all', inplace=True)
mySelection_.dropna(subset=['deaths'], how='all', inplace=True)
mySelection_.shape
mySelection_
for item_ in ['confirmed','delta_confirmed','alternative_source','link']:
    try:
        mySelection_.drop([item_], axis=1, inplace=True)
    except:
        pass
# drop duplicated rows:
mySelection_.drop_duplicates(keep='last', inplace=True)
# drop columns with nans:
mySelection_.dropna(axis=1, how='all', inplace=True)
print('mySelection_: shapes = ', mySelection_.shape)   
all_cols_ = mySelection_.columns.tolist()
check_cols_ = list(set(all_cols_) - set(['Country','year','month','day','WeekOfYear','DayOfWeek','alternative_source','link']))

mySelection_.dropna(subset=check_cols_, how='all', inplace=True)
print ('mySelection_.shape = ', mySelection_.shape)
# group 1
myDirs = ['world_bank_group1']

def recreate_row_group1(Selected_row):
     
    numerical_cols_ = Selected_row.columns[Selected_row.columns.str.isnumeric()].tolist()
    
    new_df_ = pd.DataFrame(columns=['Country', 'country_code', 'indicator_code', ])
    new_df_['Country'] = [Selected_row.Country.values[0]]
    new_df_['country_code'] = Selected_row.country_code.values[0]
    code_ = Selected_row.indicator_code.values[0]
    new_df_['indicator_code'] =  code_ + '__' + Selected_row.indicator_name.values[0]
    for item_ in numerical_cols_:
        new_col_= item_ + '_' + code_
        new_df_[new_col_] = Selected_row[item_].values[0]
        
    return new_df_
Merge_list = [Key]
for dirname_ in myDirs:
  dir_ = dir_prefix + dirname_
  print ('start: dir = ', dir_)
  for dirname, _, filenames in os.walk(dir_):
    for filename in filenames:
        myFile = os.path.join(dirname, filename)
        print('file  = ',myFile)
        df = pd.read_csv(myFile)
        mytmp_ = get_rows(df, MyCountry, Key).copy()
        
        list_of_unique_indicator_codes = mytmp_.indicator_code.unique()
        ind_ = len(list_of_unique_indicator_codes)
        for val_ in list_of_unique_indicator_codes:
            print ('ind, Value: ',ind_, val_)
            tmp_ = mytmp_[mytmp_.indicator_code == val_].copy()
            to_add_ = recreate_row_group1(tmp_)
            
            mySelection_ = pd.merge(mySelection_, to_add_, on=Merge_list, how='outer')
            print('mySelection_: shapes = ',mySelection_.shape)
            ind_ = ind_ - 1
                   
        
        print ('___________________')
Merge_list = [Key, 'year','month','day']
groups = mySelection_.groupby(Merge_list) 
output_ = pd.DataFrame()
ind_ = 0
for (frameno, frame) in groups:
    print ('frameno = ',frameno)
    frame.fillna(method='bfill', inplace=True)
    # keep only 1st row in the group:
    output_ = output_.append(frame.head(1))
    ind_ += 1
    print ('=====================')
item_ = 'cases'
AUC_ = np.trapz(np.array(output_[item_].values), dx = 1)
output_['AUC__'+item_] = AUC_
print (AUC_)
item_ = 'deaths'
AUC_ = np.trapz(np.array(output_[item_].values), dx = 1)
output_['AUC__'+item_] = AUC_
print (AUC_)

# ratio: deaths/cases
eps = 1.
ratio=np.array(output_.deaths.values)/(np.array(output_.cases.values) + eps)
AUC_ = np.trapz(ratio, dx = 1)
print (AUC_)
output_['AUC__ratio_death_to_cases'] = AUC_
output_.dropna(axis=1, how='all',inplace=True)
output__ = output_.copy()
Col_to_delete = ['day','month','year','cases','deaths','DayOfWeek','WeekOfYear','combined_key','x','y','iso']
for item_ in Col_to_delete:
    try:
        output__.drop([item_], axis=1, inplace=True)
    except:
        pass
# delete duplicated rows:
output__.drop_duplicates(keep='last', inplace=True)
# delete columns with all nan:
output__.dropna(axis=1, how='all',inplace=True)
output__.shape
# show columns with not the same  values:
frame_ = output__.loc[:, ~(output__ == output__.iloc[0]).all()].copy()
print (frame_)
# here we need some hand's work to do:

## some other possibilities:
#
#SelCols = [ 'total_confirmed_cases_of_covid_19_cases',
#           'total_covid_19_tests_per_million_people',
#           'total_confirmed_cases_of_covid_19_per_million_people_cases_per_million']
#
#SelCols = ['region_type','percent_yoy_change','sources','info',
#           'total_confirmed_cases_of_covid_19_cases','total_covid_19_tests_per_million_people',
#'total_confirmed_cases_of_covid_19_per_million_people_cases_per_million']
#
#SelCols = ['total_covid_19_tests_per_million_people',
#            'total_confirmed_cases_of_covid_19_per_million_people_cases_per_million']
#
SelCols = ['info',
           'sources',
            #'code',
            #'percent_yoy_change',
            #'region_type',
           'total_confirmed_cases_of_covid_19_cases', 
           'total_covid_19_tests_per_million_people',
            'total_confirmed_cases_of_covid_19_per_million_people_cases_per_million']

for c_ in SelCols:
    #print ('1:',c_)
    value_ = output__[output__[c_].notnull()][c_].values[0]
    object_ = output__[c_].dtype
    #print (c_, value_, object_)
    output__[c_] = output__[c_].astype(object_).fillna(value_) 
    #print ('AFTER : c_, ... = \n',c_, value_, output__[c_].unique().tolist())
    #print ('===============')

output__ = output__[output__['scale'].notna()]
#output__ = output__[output__['code'].notna()]
#output__ = output__[output__['sources'].notna()]
#output__ = output__[output__['source_type'].notna()]
#output__ = output__[output__['scale'].notna()]
#output__ = output__[output__['info'].notna()]
#output__ = output__[output__['total_confirmed_cases_of_covid_19_cases'].notna()]
# show columns with not the same  values:
frame_ = output__.loc[:, ~(output__ == output__.iloc[0]).all()].copy()
print (frame_)
output__.shape, output_.shape
# save data - time dependent part
Filename_ = './Q1__' + MyCountry + '_p1.csv'
print (Filename_)
output_.to_csv(Filename_,index=False)
# save data - single row per country
Filename_ = './Q1__' + MyCountry + '_p2_stat.csv'
print (Filename_)
output__.to_csv(Filename_,index=False)
