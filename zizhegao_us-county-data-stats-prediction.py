# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np



from sklearn.metrics import roc_curve,auc

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, KBinsDiscretizer



import matplotlib.pyplot as plt

import seaborn as sns



# show all columns of dataframe

pd.set_option('display.max_columns', None)
# read counties dataset

orig_df = pd.read_csv('/kaggle/input/covid19-in-usa/us_counties_covid19_daily.csv')



proc_df = orig_df.copy()



# dataset size

print('df shape: {}'.format(proc_df.shape))

proc_df.head(5)
# 数据集中最后统计的时间

print('lattest date in dataset: {}'.format(proc_df.sort_values(by=['date'], ascending=False)['date'].max()))

print('earliest date in dataset: {}'.format(proc_df.sort_values(by=['date'], ascending=False)['date'].min()))

print('total states number: {}'.format(len(proc_df.state.unique())))

print('total counties number: {}'.format(len(proc_df[proc_df['date'] == '2020-07-20']['county'])))

print('total unique counties number(different state may have same county name): {}'.format(len(proc_df.county.unique()))) # 不同州的county的名字是有重复的
# As the cases and deaths number is accumulated, so we just use the last day of month

def summarize_total_numbers(df, date):

    tmp_df = df.copy()

    

    final_df = tmp_df[tmp_df['date'] == date]

    print('total cases: {}'.format(final_df.cases.sum()))

    print('total deaths: {}'.format(final_df.deaths.sum()))
summarize_total_numbers(df=proc_df, date='2020-07-20')
def get_max_min_cases_deaths_per_state_county(df, date):

    tmp_df = df.copy()

    

    final_df = tmp_df[tmp_df['date'] == date]

        

    # most cases/deaths of states

    max_case_state_df = final_df.groupby('state')['cases'].apply(sum).sort_values(ascending=False).reset_index()

    max_death_state_df = final_df.groupby('state')['deaths'].apply(sum).sort_values(ascending=False).reset_index()

    print('max case number of states - {}: {}'.format(max_case_state_df.iloc[0]['state'], max_case_state_df.iloc[0]['cases']))

    print('max death number of states - {}: {}'.format(max_death_state_df.iloc[0]['state'], max_death_state_df.iloc[0]['deaths']))

    

    # least cases/deaths of states

    min_case_state_df = final_df.groupby('state')['cases'].apply(sum).sort_values(ascending=True).reset_index()

    min_death_state_df = final_df.groupby('state')['deaths'].apply(sum).sort_values(ascending=True).reset_index()

    print('min case number of states - {}: {}'.format(min_case_state_df.iloc[0]['state'], min_case_state_df.iloc[0]['cases']))

    print('min death number of states - {}: {}'.format(min_death_state_df.iloc[0]['state'], min_death_state_df.iloc[0]['deaths']))

    

    # most cases/deaths of counties

    max_cases_county_df = final_df.sort_values(by=['cases'], ascending=False).head(1)

    max_deaths_county_df = final_df.sort_values(by=['deaths'], ascending=False).head(1)

    print('state: {}, county: {}, max cases: {}'.format(max_cases_county_df.iloc[0]['state'], max_cases_county_df.iloc[0]['county'], max_cases_county_df.iloc[0]['cases']))

    print('state: {}, county: {}, max deaths: {}'.format(max_deaths_county_df.iloc[0]['state'],max_deaths_county_df.iloc[0]['county'], max_deaths_county_df.iloc[0]['deaths']))

    

    # least cases/deaths of counties

    min_cases_county_df = final_df.sort_values(by=['cases'], ascending=True).head(1)

    min_deaths_county_df = final_df.sort_values(by=['deaths'], ascending=True).head(1)

    print('state: {}, county: {}, min cases: {}'.format(min_cases_county_df.iloc[0]['state'], min_cases_county_df.iloc[0]['county'], min_cases_county_df.iloc[0]['cases']))

    print('state: {}, county: {}, min deaths: {}'.format(min_deaths_county_df.iloc[0]['state'], min_deaths_county_df.iloc[0]['county'], min_deaths_county_df.iloc[0]['deaths']))

get_max_min_cases_deaths_per_state_county(df=proc_df, date='2020-07-20')
def avg_and_median_of_state_county_cases_deaths(df, date):

    tmp_df = df.copy()

    final_df = tmp_df[tmp_df['date'] == date]

    

    gb_df = final_df.groupby('state')['cases','deaths'].apply(sum).reset_index()

    

    # avg/median of states

    print('states avg cases: {}, median cases: {}'.format(gb_df.cases.mean(), gb_df.cases.median()))

    print('states avg deaths: {}, median cases: {}'.format(gb_df.deaths.mean(), gb_df.deaths.median()))

    

    # avg/median of counties

    print('mean cases of each state:')

    print(final_df.groupby('state')['cases','deaths'].mean().reset_index().sort_values(by=['cases'], ascending=False))

    

    print('median cases of each state:')

    print(final_df.groupby('state')['cases','deaths'].median().reset_index().sort_values(by=['cases'], ascending=False))
avg_and_median_of_state_county_cases_deaths(df=proc_df, date='2020-07-20')
def each_month_increment_cases_deaths_of_states(df):

    tmp_df = df.copy()

    

    def get_last_day_of_month():

        from pandas.tseries.offsets import MonthEnd

        dates = (pd.to_datetime(tmp_df['date']) + MonthEnd(1)).dt.strftime('%Y-%m-%d').unique()

        return dates

    

    last_dates = get_last_day_of_month()

    

    # replace 7-31 with 7-20

    last_dates = np.append(last_dates, ['2020-07-20'])

    

    # get final stats of each state and county by selecting data with last day

    month_stats_df = tmp_df[tmp_df['date'].isin(last_dates)]

    month_stats_df['month'] = pd.to_datetime(month_stats_df['date'], format='%Y-%m').dt.strftime('%Y-%m')

    

    

    state_month_df = month_stats_df.groupby(['state', 'month']).sum().reset_index()

    

    # iterate each state, output diff figures of each state in 7 months

    for state in tmp_df['state'].unique():

        state_df = state_month_df[state_month_df['state']==state].sort_values(by=['month'], ascending=True)

        

        # cases/death diff

        state_df['case_diff'] = state_df['cases'].diff()

        state_df['death_diff'] = state_df['deaths'].diff()

        

        print('state month cases/death diff: {}'.format(state))

        print(state_df)
each_month_increment_cases_deaths_of_states(proc_df)


# get the severe counties of fed & state.

def severe_county_by_month(df):

    tmp_df = df.copy()

    

    last_dates = ['2020-05-31', '2020-06-30','2020-07-20']

    

    # get data with the last day filter

    month_stats_df = tmp_df[tmp_df['date'].isin(last_dates)]

    month_stats_df['month'] = pd.to_datetime(month_stats_df['date'], format='%Y-%m').dt.strftime('%Y-%m')

    

    # As county in different states may have the same name, so we combine State and County name to distinguish.

    month_stats_df['new_county'] = month_stats_df['state'] + '_' + month_stats_df['county']

    

    county_month_df = month_stats_df.groupby(['new_county', 'month']).sum().reset_index()

    

    # all counties in the Federal

    county_list = county_month_df['new_county'].unique()

    

    # the avg incr of all counties in the federal

    fed_county_diff=[]

    for cty in county_list:

        cty_df = county_month_df[county_month_df['new_county']==cty]

        

        # there is no such situation that no records in previous month but have records in latter month

        if len(cty_df) < 3:

            continue

        

        cases_diff = cty_df['cases'].reset_index().diff()['cases'][2]

        

        fed_county_diff.append(cases_diff)

    fed_avg_cases_diff = sum(fed_county_diff)/len(fed_county_diff)

    print('fed avg cases incr: {}'.format(fed_avg_cases_diff))

    

    # avg incr of counties of states

    state_county_diff_sum={}

    for cty in county_list:

        cty_df = county_month_df[county_month_df['new_county']==cty]

        

        if len(cty_df) < 3:

            continue

            

        cases_diff = cty_df['cases'].reset_index().diff()['cases'][2]

        

        arr = cty.split('_')

        

        state_name=arr[0]

        cty_name=arr[1]

        

        if state_name in state_county_diff_sum:

            cty_value = state_county_diff_sum[state_name]

            state_county_diff_sum[state_name] = cty_value + float(cases_diff)

        else:

            state_county_diff_sum[state_name] = float(cases_diff)

            

    state_county_number_dict = tmp_df[['state','county']].drop_duplicates().groupby(['state']).count().to_dict()['county']

    

    state_county_diff={}

    for state in state_county_diff_sum:

        cty_sum = state_county_diff_sum[state]

        num=state_county_number_dict[state]

        

        state_county_diff[state]=cty_sum/num

    

    print('state avg cases incr: {}'.format(state_county_diff))

        

    

    severe_county_fed = []

    severe_county_state = []

    for cty in county_list:

        cty_df = county_month_df[county_month_df['new_county']==cty]

        

        if len(cty_df) < 3:

            continue

        

        cases_diff = cty_df['cases'].reset_index().diff()

        deaths_diff = cty_df['deaths'].reset_index().diff()

        



        # incr larger than avg incr of federal

        if cases_diff['cases'][2] >= cases_diff['cases'][1] and deaths_diff['deaths'][2] >= deaths_diff['deaths'][1] and cases_diff['cases'][2] >= fed_avg_cases_diff:

            severe_county_fed.append(cty)

            

        # incr larger than avg incr of states

        arr = cty.split('_')

        

        state_name=arr[0]

        cty_name=arr[1]

        state_cty_incr=state_county_diff[state_name]

        if cases_diff['cases'][2] >= cases_diff['cases'][1] and deaths_diff['deaths'][2] >= deaths_diff['deaths'][1] and cases_diff['cases'][2] >= state_cty_incr:

            severe_county_state.append(cty)

            

        

    print('severe counties count of fed: {}, list: {}'.format(len(severe_county_fed), severe_county_fed))

    print('severe counties count of state: {}, list: {}'.format(len(severe_county_state), severe_county_state))
severe_county_by_month(df=proc_df)
# create train set 

def build_train_set(df):

    tmp_df=df.copy()

    

    def get_last_day_of_month():

        from pandas.tseries.offsets import MonthEnd

        dates = (pd.to_datetime(tmp_df['date']) + MonthEnd(1)).dt.strftime('%Y-%m-%d').unique()

        return dates

    

    last_dates = get_last_day_of_month()

    

    last_dates = np.append(last_dates, ['2020-07-20'])

    # last_dates.append(['2020-07-20'])

    

    month_stats_df = tmp_df[tmp_df['date'].isin(last_dates)]

    month_stats_df['month'] = pd.to_datetime(month_stats_df['date'], format='%Y-%m').dt.strftime('%Y-%m')

    

    

    state_month_df = month_stats_df.groupby(['state', 'month']).sum().reset_index()

    

    train_df = pd.DataFrame(columns=['state','m2_c','m2_d','m3_c','m3_d','m4_c','m4_d','m5_c','m5_d','m6_c','m6_d','m7_c','m7_d','future'])

    

    month_list = ['2020-01','2020-02','2020-03','2020-04','2020-05','2020-06','2020-07']

    

    # iterate each state, output diff of cases and deaths of each state during 7 months.

    state_train_row=[]

    for state in tmp_df['state'].unique():

        state_df = state_month_df[state_month_df['state']==state].sort_values(by=['month'], ascending=True)

        

        # cases/death diff

        state_df['case_diff'] = state_df['cases'].diff()

        state_df['death_diff'] = state_df['deaths'].diff()

        

        cases_month_list = state_df['month'].unique()

        

        lst = list(set(month_list) - set(cases_month_list))

        

        if len(lst) == 0:

            pass

        else:

            for m in lst:

                state= state_df.iloc[0][0]

                new_df = pd.DataFrame([[state, m, 0, 0,0,0,0]], columns=['state','month','fips ','cases','deaths','case_diff','death_diff'])

                state_df = pd.concat([state_df,new_df], axis=0)

            

        state_sort_df = state_df.sort_values(by=['month'], ascending=True)

        #print(state_sort_df)

        train_row = []

        for row in state_sort_df.itertuples():

            

            state = row.state

            month = row.month

            case_diff = row.case_diff

            death_diff = row.death_diff

            

            if len(train_row) == 0:

                train_row.append(state)

            

            train_row.append(case_diff)

            train_row.append(death_diff)

        

        state_train_row.append(train_row)

        

#         print('train row:')

#         print(train_row)

        

#         print('state train row:')

#         print(state_train_row)

        

    return pd.DataFrame(state_train_row, columns=['state','m1_c','m1_d','m2_c','m2_d','m3_c','m3_d','m4_c','m4_d','m5_c','m5_d','m6_c','m6_d','m7_c','m7_d'])

train_df = build_train_set(df=proc_df)

train_df
# if incr of cases and deaths both go down, we set label as 1 for this sample, it means the tendency is controlled. Otherwise,

# the label of this samples will be set to 0, which means the tendency keeps spreading.

def set_label(row):

    if row['m6_c']>=row['m7_c'] and row['m6_d']>=row['m7_d']:

        return 1

    else:

        return 0
train_df['label']=train_df.apply(set_label, axis=1)

train_df
train_set = train_df.copy()

train_set = train_set.fillna(0)
# generate X and y 

y=train_set.pop('label')

X=train_set



states = X.pop('state')

states
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_validate

from sklearn import linear_model



#lr=LogisticRegression(random_state=0)

lr=linear_model.RidgeClassifier()

scores = cross_validate(lr,X, y, cv=10, return_estimator=True, return_train_score=True, scoring=['accuracy', 'roc_auc'])

scores
print("AUC: %0.2f (+/- %0.2f)" % (scores['train_roc_auc'].mean(), scores['train_roc_auc'].std() * 2))
cls_model = scores.get('estimator')[1]

cls_model
predict_state = cls_model.predict(X)

state_tendency_df = pd.concat([pd.DataFrame(states, columns=['state']),pd.DataFrame(predict_state, columns=['tendency'])], axis=1)

state_tendency_df
spreading_states = state_tendency_df[state_tendency_df['tendency']==0]['state'].unique()

controlled_states = state_tendency_df[state_tendency_df['tendency']==1]['state'].unique()



print('keep spreading states, count: {}, state list:{}'.format(len(spreading_states), spreading_states))

print('controlled states, count: {}, state list:{}'.format(len(controlled_states), controlled_states))