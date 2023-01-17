# This notebook explores the Kickstarter data set.
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

sns.set(style="whitegrid")

sns.set_context(context=None, font_scale=1, rc=None)

plt.rcParams['xtick.labelsize']=16

plt.rcParams['ytick.labelsize']=16

plt.rcParams['axes.titlesize']=32
# Reading in data from csv



kickstarter = pd.read_csv('../input/kickstarter.csv', index_col = 0)
# Taking a look at the first few rows.



kickstarter.head()
# How many Kickstarter projects are there? What percentage of them were successful?

original_success = kickstarter.groupby(by='state')['goal'].count()

new_state_success = kickstarter.groupby(by='binary_state')['goal'].count()



original_success_labels = original_success.index.values.tolist()

original_success_sizes = original_success.values.tolist()



new_state_labels = new_state_success.index.values.tolist()

new_state_sizes = new_state_success.values.tolist()

f, ax = plt.subplots(figsize=(8, 8))

sns.barplot(x=original_success_sizes, y=original_success_labels)

sns.despine(trim=True)

ax.set_title("Number of Kickstarter Projects by Category")

ax.set_xlabel("Number of Projects")

plt.show()
new_state_labels = new_state_success.index.values.tolist()

new_state_sizes = new_state_success.values.tolist()



f, ax = plt.subplots(figsize=(8, 8))

plt.pie(new_state_sizes, labels=new_state_labels, autopct='%.0f%%', labeldistance=.5)

plt.title("Adjusted for Binary Success / Failure")

plt.show()

failed = kickstarter[kickstarter['binary_state'] == 'failed']['goal_USD']

successful = kickstarter[kickstarter['binary_state'] == 'successful']['goal_USD']

bins = pd.qcut(failed, 10, retbins=True, precision=0)



by_goal = kickstarter.groupby(pd.qcut(kickstarter.goal_USD, 10))['binary_state'].value_counts()

by_goal = by_goal.unstack(['binary_state'])

by_goal['success_rate'] = by_goal['successful'] / (by_goal['successful'] + by_goal['failed'])

by_goal['bins'] = ['Lowest Decile (< $600)', '$600-$1,322', '$1,322-$2,500', '$2,500-$3,800', 

                      '$3,800-$5,000', '$5,000-$8,000', '$8,000-$12,000', '$12,000-$20,000', 

                       '$20,000-$40,000', 'Highest decile (> $40,000)']



f, ax = plt.subplots(figsize=(12, 8))

by_goal['success_rate'].plot(kind='bar')

ax.set_title('Smaller Goals for More Success')

ax.set_xlabel('Goal Decile')

ax.set_ylabel('Success Rate')

ax.set_xticklabels(by_goal['bins'])

xvals = ax.get_xticks()

yvals = ax.get_yticks()

f.autofmt_xdate()

sns.despine(trim=True)

ax.set_yticklabels(['{:,.2%}'.format(x) for x in yvals])

plt.show()

livebymonth = kickstarter[kickstarter['state'] == 'live'].groupby(['year', 'month']).count()

f, ax = plt.subplots(figsize=(8, 8))

livebymonth['id'].plot()

ax.set_title('When "Live" Projects Launched')

xvals = ax.get_xticks()

yvals = ax.get_yticks()

f.autofmt_xdate()
deadlinedays = kickstarter[kickstarter['state'] == 'live'].groupby('days_to_deadline').count()

f, ax = plt.subplots(figsize=(8, 8))

ax.set_title("Days to Deadline for 'Live' Projects")

deadlinedays['id'].plot()

plt.show()
# Breaking down success by category slug

success_by_category_name = pd.pivot_table(kickstarter, index='category_slug', columns='binary_state', values='goal', aggfunc='count', fill_value = 0)

print(success_by_category_name)
# Breaking down success by category

success_by_category_name = pd.pivot_table(kickstarter, index='category_slug', columns='binary_state', values='goal', aggfunc='count', fill_value = 0)

success_by_category_name['total'] = success_by_category_name['failed'] + success_by_category_name['successful'] 

success_by_category_name['success_rate'] = success_by_category_name['successful'] / success_by_category_name['total']

topcategories = success_by_category_name.sort_values(by='category_slug').reset_index()



f, ax = plt.subplots(figsize=(8, 8))



sns.barplot(x="total", y="category_slug", data=topcategories, label="Failed", color="b")

sns.barplot(x="successful", y="category_slug", data=topcategories, label="Successful", color='orange')

sns.despine(trim=True)



plt.title("Success by Category")

plt.xlabel("Number of Projects")

plt.ylabel("Category")

plt.legend()

plt.show()

# Top Subcategories



# Breaking down success by subcategory

success_by_subcategory = pd.pivot_table(kickstarter, index='category_name', columns='binary_state', values='goal', aggfunc='count', fill_value = 0)

success_by_subcategory['total'] = success_by_subcategory['failed'] + success_by_subcategory['successful'] 

success_by_subcategory['success_rate'] = success_by_subcategory['successful'] / success_by_subcategory['total']

topcategories = success_by_subcategory.sort_values(by='total', ascending=False).reset_index().head(20)



f, ax = plt.subplots(figsize=(8, 8))



sns.barplot(x='total', y='category_name', data=topcategories, label='Failed', color="b")

sns.barplot(x='successful', y='category_name', data=topcategories, label='Successful', color='orange')

sns.despine(trim=True)



plt.title("Success by Subcategory")

plt.xlabel("Number of Projects")

plt.ylabel("Subcategory")

plt.legend()

plt.show()

# Top Subcategories



# Breaking down success by subcategory

success_by_subcategory = pd.pivot_table(kickstarter, index='category_name', columns='binary_state', values='goal', aggfunc='count', fill_value = 0)

success_by_subcategory['total'] = success_by_subcategory['failed'] + success_by_subcategory['successful'] 

success_by_subcategory['success_rate'] = success_by_subcategory['successful'] / success_by_subcategory['total']

topcategories = success_by_subcategory.sort_values(by='success_rate', ascending=False).reset_index().head(20)



f, ax = plt.subplots(figsize=(8, 8))



sns.barplot(x='total', y='category_name', data=topcategories, label='Failed', color="b")

sns.barplot(x='successful', y='category_name', data=topcategories, label='Successful', color='orange')

sns.despine(trim=True)



plt.title("Success by Subcategory")

plt.xlabel("Number of Projects")

plt.ylabel("Subcategory")

plt.show()
f, ax = plt.subplots(figsize=(8, 8))

sns.boxplot(x='backers_count', y='category_slug', data=kickstarter, whis=[25, 75])

ax.set_xscale("log")

ax.set_title("Number of Backers by Category")

ax.set_xlabel("Number of Backers", fontsize=16)

ax.set_ylabel("Category", fontsize=16)

plt.show()
# How many projects have N number of backers? If you have N backers, what is your success rate?

nbackers = kickstarter.groupby(['backers_count', 'binary_state']).count()

nbackers = nbackers['state'].unstack()

nbackers = nbackers.fillna(0)

nbackers['success_rate'] = nbackers['successful'] / (nbackers['failed'] + nbackers['successful'])

plot = nbackers['success_rate'][0:500]

nbackersdf = pd.DataFrame(nbackers[0:501])



f, ax = plt.subplots(figsize=(12, 8))



sns.scatterplot(data = nbackersdf['success_rate'])

ax.set_xlabel('Number of Backers')

ax.set_ylabel('Success Rate')

ax.set_title('Number of Backers Needed for Success')



vals = ax.get_yticks()

ax.set_yticklabels(['{:,.2%}'.format(x) for x in vals])

plt.show()

# Zero followers

kickstarter[kickstarter['backers_count'] == 0].groupby('category_slug').count()['id'] 

zero_followers = kickstarter[kickstarter['backers_count'] == 0].groupby('category_slug').count()['id']  / kickstarter['category_slug'].value_counts()

f, ax = plt.subplots(figsize=(8, 8))

zero_followers.plot(kind='bar')

f.autofmt_xdate()

ax.set_title("Percentage of Projects with Zero Backers")

vals = ax.get_yticks()

ax.set_yticklabels(['{:,.2%}'.format(x) for x in vals])

plt.show()
f, ax = plt.subplots(figsize=(8, 8))

g = sns.violinplot(x=kickstarter["staff_pick"], y=np.log1p(kickstarter["usd_pledged"]), hue=kickstarter["binary_state"], cut=0)

vals = ax.get_yticks()

exp_vals = [np.exp(val) for val in vals]

ax.set_yticklabels(['${0:7.2f}'.format(x) for x in exp_vals])

g.set_title("Amount Raised Depending on Staff Pick")

plt.show()
byspotlight = kickstarter.groupby(by=['spotlight', 'binary_state']).count()

byspotlight = byspotlight['id'].unstack('binary_state').fillna(0)

byspotlight['success_rate'] = byspotlight['successful'] / (byspotlight['successful'] + byspotlight['failed'])



bystaffpick = kickstarter.groupby(by=['staff_pick', 'binary_state']).count()

bystaffpick = bystaffpick['id'].unstack('binary_state').fillna(0)

bystaffpick['success_rate'] = bystaffpick['successful'] / (bystaffpick['successful'] + bystaffpick['failed'])



f, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, figsize=(8, 8))

f.autofmt_xdate()

ax1.set_title('Success Rate by Spotlight', fontsize=16)

ax2.set_title('Success Rate by Staff Pick', fontsize=16)



byspotlight

byspotlight['success_rate'].plot(kind='bar', ax=ax1)

bystaffpick['success_rate'].plot(kind='bar', ax=ax2)



plt.show()
timeseries = kickstarter.groupby(['year', 'month', 'binary_state']).count()

timeseries = timeseries['id'].unstack('binary_state').fillna(0)

timeseries['success_rate'] = timeseries['successful'] / (timeseries['failed'] + timeseries['successful'])



f, ax = plt.subplots(figsize=(12, 8))

timeseries['success_rate'].plot()

ax.set_title('Project success over time')

xvals = ax.get_xticks()

yvals = ax.get_yticks()

ax.set_yticklabels(['{:,.2%}'.format(x) for x in yvals])

f.autofmt_xdate()
# Success rate by US state

byusstate = kickstarter[kickstarter['location_country']=='US'].groupby(['location_state', 'binary_state']).count()

byusstate.drop(['Erongo', 'Khomas', 'Okavango', '-'], inplace=True)

byusstate = byusstate['id'].unstack('binary_state')

byusstate['success_rate'] = byusstate['successful'] / (byusstate['failed'] + byusstate['successful'])

print(byusstate.sort_values('success_rate', ascending=False).head(5))
# Success rate by country

byworld = kickstarter.groupby(['location_country', 'binary_state']).count()

byworld = byworld['id'].unstack('binary_state').fillna(0).sort_values('successful', ascending=False)

byworld['success_rate'] = byworld['successful'] / (byworld['successful'] + byworld['failed'])

byworld['total'] = byworld['successful'] + byworld['failed']



f, ax = plt.subplots(figsize=(8, 8))

sns.barplot(x='location_country', y='total', data=byworld[0:20].reset_index(), color='orange')

sns.barplot(x='location_country', y='successful', data=byworld[0:20].reset_index(), color='blue')

ax.set_title("Number of Projects by Country Code")

yvals = ax.get_yticks()

plt.show()
f, ax = plt.subplots(figsize=(8, 8))

sns.barplot(x='location_country', y='success_rate', data=byworld[0:20].reset_index(), color='green')

yvals = ax.get_yticks()

ax.set_yticklabels(['{:,.2%}'.format(x) for x in yvals], fontsize=16)

ax.set_title("Approval Rates for Select Countries")

plt.show()
# Success by day

byday = kickstarter.groupby(['day', 'binary_state']).count()

byday = byday['id'].unstack('binary_state').fillna(0).reset_index()

byday['success_rate'] = byday['successful'] / (byday['successful'] + byday['failed'])

byday['total'] = byday['successful'] + byday['failed']





f, ax = plt.subplots(figsize=(8, 8))

sns.barplot(x='day', y='success_rate', data=byday, color='blue')

ax.set_title("Success Rate by Day")

ax.set_yticklabels(['{:,.2%}'.format(x) for x in yvals])

yvals = ax.get_yticks()

plt.show()



kickstarter.loc[:, 'weekday'] = pd.to_datetime(kickstarter[['month', 'day', 'year']]).dt.dayofweek



# Success by weekday

byweekday = kickstarter.groupby(['weekday', 'binary_state']).count()

byweekday = byweekday['id'].unstack('binary_state').fillna(0).reset_index()

byweekday['success_rate'] = byweekday['successful'] / (byweekday['successful'] + byweekday['failed'])

byweekday['total'] = byweekday['successful'] + byweekday['failed']



f, ax = plt.subplots(figsize=(12, 12))

sns.barplot(x='weekday', y='success_rate', data=byweekday)

ax.set_title("Success Rate by Weekday")

ax.set_yticklabels(['{:,.2%}'.format(x) for x in yvals])

ax.set_xticklabels(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

yvals = ax.get_yticks()

f.autofmt_xdate()

plt.show()