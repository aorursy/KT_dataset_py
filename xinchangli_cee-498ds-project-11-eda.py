# Load useful modules
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
# Print all files in the input directory (auto-generated code from Kaggle)
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Read in the .csv files as Pandas DataFrame (this can take around 2 minutes)
bldg_meta = pd.read_csv('/kaggle/input/ashrae-energy-prediction/building_metadata.csv')

train = pd.read_csv('/kaggle/input/ashrae-energy-prediction/train.csv')
weather_train = pd.read_csv('/kaggle/input/ashrae-energy-prediction/weather_train.csv')

test = pd.read_csv('/kaggle/input/ashrae-energy-prediction/test.csv')
weather_test = pd.read_csv('/kaggle/input/ashrae-energy-prediction/weather_test.csv')

sample_sub = pd.read_csv('/kaggle/input/ashrae-energy-prediction/sample_submission.csv')
# Plots formatter (borrowed from https://stackoverflow.com/questions/3899980/how-to-change-the-font-size-on-a-matplotlib-plot)
SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
N, d = bldg_meta.shape
print(f'In bldg_meta, there are {N} samples and {d} features.')
features = bldg_meta.columns
print(f'The {d} features are: ', list(features))
print('Here are the first 10 rows of the DataFrame:')
bldg_meta.head(10)
assert np.array_equal(bldg_meta.site_id.unique(), np.arange(0, 16))
assert np.array_equal(bldg_meta.building_id.unique(), np.arange(0, 1449))
print('Missing data')
for col in bldg_meta['primary_use year_built square_feet floor_count'.split()].columns:
    missing = pd.isnull(bldg_meta[col]).sum() # returns the total no. of empty entries in the column
    pct = round(missing/N*100, 1) # N is the total no. of entries = total no. of buildings
    print(f'\t{col}: \t{missing} ({pct}%)')
# Set up subplot
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
n, bins, patches = ax.hist(bldg_meta.site_id, bins=15)

# Annotate each bar with the no. of buildings in that site:
for number, b in zip(n, bins[:-1]):
    ax.annotate(int(number), 
                 xy=(b+.5, number), xytext=(0, 1),#1 point vertical offset
                 textcoords="offset points",
                 ha='center', va='bottom', fontsize=12)
# Set x ticks at the center of each bar
ax.set_xticks(np.arange(0.5, 15., step=1))
ax.set_xticklabels(np.arange(0, 16))
# Set y limits
ax.set_ylim([0, 300])
# Set x and y labels and add a title
ax.set_xlabel('site_id')
ax.set_ylabel('number of buildings')
ax.set_title('Number of buildings in each site', fontsize=16);
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for i, col in enumerate('primary_use year_built square_feet floor_count'.split()):
    bldg_meta[col].hist(xrot=90, ax=axes[i], bins=min(25, len(bldg_meta[col].unique())))
    axes[i].set_title(col)
fig.suptitle('Distributions of bldg_meta features');
bldg_meta['year_built square_feet floor_count'.split()].corr()
print('train shape: ', train.shape)
print('Missing meter_reading: ', end='')

missing = pd.isnull(train['meter_reading']).sum() # returns the total no. of empty entries in the column
pct = round(missing/N*100, 1) # N is the total no. of entries = total no. of buildings
print(f'\t{missing} ({pct}%)')

train.head(10)
# Compute the sum of meter readings
tot_meter_per_bldg = train[['building_id', 'meter_reading']].groupby('building_id').sum()
# Join with bldg_meta based on building_id
tot_meter_per_bldg = bldg_meta.merge(tot_meter_per_bldg, on='building_id')
tot_meter_per_bldg.rename(columns={'meter_reading': 'tot_meter_reading'}, inplace=True)
tot_meter_per_bldg.head(10)
# Make the plot
tot_meter_per_bldg.tot_meter_reading.plot(logy=True, style='.', 
                                          title='Sum of meter readings per building in the training set', 
                                          xlabel='building_id', ylabel='total meter reading',
                                          figsize=(10, 5));
min_id = tot_meter_per_bldg.tot_meter_reading.sort_values(ascending=True).index[0]
max_id = tot_meter_per_bldg.tot_meter_reading.sort_values(ascending=True).index[-1]
tot_meter_per_bldg[(tot_meter_per_bldg.index==min_id)|(tot_meter_per_bldg.index==max_id)]
tot_meter_per_bldg[(tot_meter_per_bldg.index!=min_id)
                   &(tot_meter_per_bldg.index!=max_id)][['square_feet', 'year_built', 'floor_count', 'tot_meter_reading']].corr()
# Join train with bldg_meta based on building_id
train_meta = bldg_meta.merge(train, on='building_id')
train_meta['timestamp'] = pd.to_datetime(train_meta.timestamp)
train_meta.head(10)
# Get a list of primary uses and its length
prim_use_list = train_meta['primary_use'].unique()
len(prim_use_list)
# Group by primary use and plot time series profiles
fig, axes = plt.subplots(8, 2, figsize=(20, 35))

# For education buildings, we will drop the two outliers we identified earlier for now; explanation comes later. 
edu_df = train_meta[(train_meta['primary_use']=='Education')&(train_meta['building_id']!=min_id)&(train_meta['building_id']!=max_id)]
# Daily energy use for each building
edu_daily = edu_df.groupby(['building_id', edu_df['timestamp'].dt.date])['meter_reading'].sum()
edu_daily = edu_daily.reset_index()
edu_mean = edu_daily.groupby('timestamp')['meter_reading'].mean()
axes[0, 0].plot(edu_mean.index, edu_mean)
axes[0, 0].set_title('Education (excl. 1099 & 740)')

# For the rest of the building types we will write a loop for batch ploting:
for ax, use in zip(axes.flat[1:], prim_use_list[1:]): 
    prim_use_df = train_meta[train_meta['primary_use']==use]
    prim_use_daily = prim_use_df.groupby(['building_id', prim_use_df['timestamp'].dt.date])['meter_reading'].sum()
    prim_use_daily = prim_use_daily.reset_index()
    mean = prim_use_daily.groupby('timestamp')['meter_reading'].mean()
    
    ax.plot(mean.index, mean)
    ax.set_title(use)

# The following lines help create common X and Y labels.
# Borrowed from: https://stackoverflow.com/questions/6963035/pyplot-axes-labels-for-subplots/36542971#36542971
fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axes
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.grid(False)
plt.xlabel('Time')
plt.ylabel('Meter Reading (Daily Sum)', labelpad=20)

plt.title('Time series profiles for different building types', pad=30)
plt.show()
train_meta['month'] = train_meta['timestamp'].dt.month
# Outliers: 
max_edu_daily = train_meta[train_meta['building_id']==max_id].groupby(train_meta['timestamp'].dt.date)['meter_reading'].sum()
min_edu_daily = train_meta[train_meta['building_id']==min_id].groupby(train_meta['timestamp'].dt.date)['meter_reading'].sum()

fig, axes = plt.subplots(1, 2, figsize=(18, 4))
axes[0].plot(max_edu_daily.index, max_edu_daily)
axes[0].set_title(f'maximum consumption ({max_id})')
axes[1].plot(min_edu_daily.index, min_edu_daily)
axes[1].set_title(f'minimum consumption ({min_id})')

fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axes
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.grid(False)
plt.xlabel('Time')
plt.ylabel('Meter Reading (Daily Sum)', labelpad=20)

plt.title('Time series profiles for the maximum- and minimum- consumption building', pad=30)
plt.show()
train_meta_no_outlier = train_meta[(train_meta['building_id']!=min_id)&(train_meta['building_id']!=max_id)]
meters = {0: 'electricity', 1: 'chilledwater', 2: 'steam', 3: 'hotwater'}

fig, axes = plt.subplots(2, 2, figsize=(20, 10))

for ax, m in zip(axes.flat, meters): 
    meter_df = train_meta_no_outlier[train_meta_no_outlier['meter']==m]
    meter_daily = meter_df.groupby(['meter', meter_df['timestamp'].dt.date])['meter_reading'].sum()
    meter_daily = meter_daily.reset_index()
    mean = meter_daily.groupby('timestamp')['meter_reading'].mean()
    
    ax.plot(mean.index, mean)
    ax.set_title(meters[m])

# The following lines help create common X and Y labels.
# Borrowed from: https://stackoverflow.com/questions/6963035/pyplot-axes-labels-for-subplots/36542971#36542971
fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axes
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.grid(False)
plt.xlabel('Time')
plt.ylabel('Meter Reading (Daily Sum)', labelpad=20)

plt.title('Time series profiles for different meter types', pad=30)
plt.show()
train_meta_no_outlier['day_of_week'] = train_meta_no_outlier['timestamp'].dt.weekday
# Visualize weekly profiles
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
for ax, m in zip(axes.flat, meters): 
    mean = train_meta_no_outlier[train_meta_no_outlier['meter']==m].groupby('day_of_week')['meter_reading'].mean()
    
    ax.plot(mean.index, mean)
    ax.set_title(meters[m])
    ax.set_xticks(np.arange(0, 7))
    ax.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])

fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axes
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.grid(False)
plt.xlabel('Weekday')
plt.ylabel('Meter Reading (Average Daily Sum )', labelpad=20)

plt.title('Time series profiles for different meter types', pad=30)
plt.show()
# Load profiles by site_id
site_daily_sum = train_meta_no_outlier.groupby(['site_id', train_meta_no_outlier.timestamp.dt.date])['meter_reading'].sum().reset_index()

fig, axes = plt.subplots(8, 2, figsize=(20, 35), tight_layout=True)
for i, ax in zip(range(16), axes.flat):
    site_daily_sum[site_daily_sum.site_id == i][['timestamp', 'meter_reading']].plot(ax = ax, x = 'timestamp', legend=False);
    ax.set_title(f'site {i}')
    
fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axes
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.grid(False)
plt.xlabel('Time')
plt.ylabel('Meter Reading (Average Daily Sum )', labelpad=20)

plt.title('Time series profiles for different meter types', pad=30)
plt.show()
weather_train['timestamp'] = pd.to_datetime(weather_train.timestamp)
weather_test['timestamp'] = pd.to_datetime(weather_test.timestamp)

print(f'weather_train: {weather_train.shape}')
display(weather_train.head())
print(f'weather_test: {weather_test.shape}')
display(weather_test.head())
def weather_missing(df, df_name):
    N = df.shape[0]
    print(f'Missing data in {df_name}:')
    for col in df.columns[2:]:
        missing = pd.isnull(df[col]).sum() # returns the total no. of empty entries in the column
        pct = round(missing/N*100, 1) # N is the total no. of entries
        print(f'\t{col}: \t{missing} ({pct}%)')
        
weather_missing(weather_train, 'weather_train')
print('')
weather_missing(weather_test, 'weather_test')
fig, axes = plt.subplots(7, 1, sharex=True, figsize=(20, 25))

for col, ax in zip(weather_train.columns[2:], axes):
    weather_train[['timestamp', col]].set_index('timestamp').resample('D').mean()[col].plot(ax=ax, color='tab:blue', label='train')
    weather_test[['timestamp', col]].set_index('timestamp').resample('D').mean()[col].plot(ax=ax, color='tab:orange', label='test')
    ax.set_ylabel(col)

axes[0].set_title('Weather data time series')
axes[-1].set_xlabel('Time')
plt.show()
train_meta_weather_no_outlier = train_meta_no_outlier.merge(weather_train, on=['site_id', 'timestamp'])
train_meta_weather_no_outlier.head()
train_meta_weather_no_outlier['hour'] = train_meta_weather_no_outlier.timestamp.dt.hour
for m in meters:
    print(meters[m])
    display(train_meta_weather_no_outlier[train_meta_weather_no_outlier.meter==m][train_meta_weather_no_outlier.columns[-8:]].corr())
