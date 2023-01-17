# Imports
import datetime
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

# Returns Panda DataFrame for provided CSV filename (without .csv extension).
load = lambda name: pd.read_csv('../input/{}.csv'.format(name))

# Information about public kernel versions. 
# Signup date table and user events table will be derived from this dataframe.
kernel_versions = load('KernelVersions')
# Convert CreationDate from string to date. It will make filtering the data easier later on.
kernel_versions['CreationDate'] = (
    pd.to_datetime(kernel_versions['CreationDate'], format='%m/%d/%Y %I:%M:%S %p', cache=True)
    .dt.normalize())
# Create 'User Signup' table
# We consider the first time someone created a kernel as their "signup" date.
kernel_users = (
    kernel_versions
    .groupby('AuthorUserId', as_index=False)
    .agg({'CreationDate': 'min'})
    .rename(columns={'AuthorUserId': 'Id', 'CreationDate': 'RegisterDate'})
)

# Assert each row represents a unique user.
assert kernel_users['Id'].nunique() == kernel_users.shape[0], 'kernel_users table is malformed.'

# Display snippet of table.
kernel_users.head()
# Create 'User Events' table.
# A user is active on a given day if there's an entry on this table for that user.
#
# If user X has 3 events on date '2018/01/01', we only need one of these events. 
# We remove extra events with the drop_duplicates function.
kernel_user_events = (
    kernel_versions[['AuthorUserId', 'CreationDate']]
    .drop_duplicates()
    .rename(columns={'AuthorUserId': 'Id', 'CreationDate': 'Date'})
)

# Display snippet of table.
kernel_user_events.head()
# Merge kernel_users (signup table) with kernel_user_events (events table).
dim_users = pd.merge(
    kernel_user_events,
    kernel_users,
    how='left',
    on='Id')

# Compute the number of weeks between signup and a given event.
dim_users['weeks_from_signup'] = round((dim_users['Date'] - dim_users['RegisterDate']) / np.timedelta64(1, 'W'))
dim_users = dim_users[['Id', 'weeks_from_signup']].drop_duplicates()
# Let's only look at the first 8 weeks after signup. 
# This is enough time for the week-over-week retention curve to converge.
dim_users = dim_users[dim_users['weeks_from_signup'] <= 8]

# Convert absolute user count each week as percentage of all users.
cohort_size = dim_users['Id'].nunique()
user_count_by_week = (
    dim_users
    .groupby('weeks_from_signup')
    .agg('count')
    .rename(columns={'Id': 'user_count'})
).reset_index()
user_count_by_week['pct_returned'] = user_count_by_week['user_count'] / cohort_size * 100

# Show retention table
user_count_by_week
ax = user_count_by_week[['weeks_from_signup', 'pct_returned']].set_index('weeks_from_signup').plot()
ax.set_ylabel('% Active Users')
_ = ax.set_title('Kernels Retention Curve')
# Load KernelTags table.
kernel_tags = load('KernelTags')
# Create temporary table to determine if a user's first kernel has a tag.
user_first_kernel = kernel_versions.iloc[kernel_versions.groupby('AuthorUserId')['CreationDate'].idxmin()]
user_first_kernel = pd.merge(
    user_first_kernel,
    kernel_tags,
    how='left',
    on='KernelId',
    suffixes=('', '_kernel_tags'))

# If right side of join is n/a, it's because user's first notebook has no tag/category.
user_first_kernel.loc[pd.notnull(user_first_kernel.TagId), 'TagId'] = 'has_category'
user_first_kernel.loc[pd.isnull(user_first_kernel.TagId), 'TagId'] = 'no_category'
user_first_kernel = user_first_kernel.rename(columns={'TagId': 'has_category'})
cohort = 'has_category'

augmented_kernel_users = pd.merge(
    kernel_users,
    user_first_kernel,
    left_on='Id',
    right_on='AuthorUserId',
    suffixes=('', '_b'))[['Id', cohort, 'RegisterDate']]

dim_users = pd.merge(
    kernel_user_events,
    augmented_kernel_users,
    how='left',
    on='Id')
dim_users['weeks_from_signup'] = round((dim_users['Date'] - dim_users['RegisterDate']) / np.timedelta64(1, 'W'))
dim_users = dim_users[['Id', 'weeks_from_signup', cohort]].drop_duplicates()
dim_users = dim_users[dim_users['weeks_from_signup'] <= 6]

assert dim_users['Id'].nunique() == dim_users[dim_users['weeks_from_signup'] == 0].shape[0]

cohort_size = (
    dim_users[dim_users['weeks_from_signup'] == 0]
    .groupby([cohort], as_index=False).agg('count')[[cohort, 'Id']]
    .rename(columns={'Id': 'cohort_size'})
)
cohort_size = cohort_size[cohort_size['cohort_size'] > 1000]


users_by_cohort = (pd.merge(
    dim_users,
    cohort_size,
    on=cohort)
 .groupby(['weeks_from_signup', cohort, 'cohort_size'], as_index=False)
 .agg('count')
 .rename(columns={'Id': 'user_count'})
)

users_by_cohort['pct'] = users_by_cohort['user_count'] / users_by_cohort['cohort_size'] * 100
plt.figure(figsize=(8, 6))
for a, b in users_by_cohort.groupby([cohort]):
    plt.plot(b['weeks_from_signup'], b['user_count'] / b['cohort_size'] * 100.0, label=a)
plt.title('Kernels Retention Curve')
plt.ylabel('% Active Users')
plt.xlabel('Weeks From Signup')
plt.legend()
plt.show()
users_by_cohort[['weeks_from_signup', 'has_category', 'pct']].pivot_table(index=['weeks_from_signup'], columns=['has_category'], values=['pct'])