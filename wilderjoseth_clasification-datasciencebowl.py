import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from scipy import stats
pd.set_option('max_colwidth', 200)
dsSpecs = pd.read_csv('../input/data-science-bowl-2019/specs.csv')
print('dsSpecs shape:', dsSpecs.shape)
dsSpecs.head()
dsSpecsArgs = pd.DataFrame()
for i in range(dsSpecs.shape[0]):
  for arg in json.loads(dsSpecs.loc[i, 'args']):
    dsSpecsArgs = dsSpecsArgs.append(pd.DataFrame({ 'event_id': dsSpecs.loc[i, 'event_id'], 'info': dsSpecs.loc[i, 'info'], 'name': arg['name'], 'type': arg['type'], 'info': arg['info'] }, index = [i]))
dsSpecsArgs.shape
dsSpecsArgs.head()
dsSpecsArgsCount =  dsSpecsArgs[['event_id', 'name']].groupby(['name']).count().reset_index()
dsSpecsArgsCount.rename(columns = { 'event_id': 'count' }, inplace = True)
dsSpecsArgsCount.sort_values(by = 'count', ascending = False, inplace = True)
print(dsSpecsArgsCount.shape)
dsSpecsArgsCount.head()
fig, ax = plt.subplots(figsize = (12, 8))
ax.barh(y = dsSpecsArgsCount.head(20)['name'], width = dsSpecsArgsCount.head(20)['count'])
ax.set_xlabel('Count')
ax.set_ylabel('Event')
plt.title('Most frequents attributes in event data')
plt.show()
del dsSpecsArgsCount
dsTrainLabels = pd.read_csv('../input/data-science-bowl-2019/train_labels.csv')
print('Shape:', dsTrainLabels.shape)
dsTrainLabels.head()
dsTrainLabels.info()
dsTotalSessions = dsTrainLabels[['accuracy_group', 'game_session']].groupby(['accuracy_group']).count().reset_index()
dsTotalSessions = dsTotalSessions.rename(columns = { 'game_session': 'count' })
dsTotalSessions.sort_values(by = 'count', ascending = False)
# Calculate number of game sessions by title
dsTrainLabels[['title', 'game_session']].groupby('title').count().reset_index()
dsTrainLabels[['title', 'accuracy_group', 
               'game_session']].groupby(['title', 'accuracy_group']).count().reset_index().pivot(index = 'title', columns = 'accuracy_group', values = 'game_session')
# Significance level
alpha = 0.05

chiSquareTest = 0
dsTrainPivot = dsTrainLabels[['title', 'accuracy_group', 
               'game_session']].groupby(['title', 'accuracy_group']).count().reset_index().pivot(index = 'title', columns = 'accuracy_group', values = 'game_session')

# Calculate chi-square test
for r in range(dsTrainPivot.values.shape[0]):
  for c in range(dsTrainPivot.values.shape[1]):
    expectedCount = dsTrainPivot.values[r, :].sum() * dsTrainPivot.values[:, c].sum() / dsTrainPivot.values.sum()
    chiSquareTest = chiSquareTest + ((dsTrainPivot.values[r, c] - expectedCount)**2) / expectedCount
print('Chi-square test:', chiSquareTest)

# Calculate degrees of freedom
df = (dsTrainPivot.shape[0] - 1) * (dsTrainPivot.shape[1] - 1)
print('Degrees of freedom', df)

# Calculate p-value
p_value = 1 - stats.chi2.cdf(chiSquareTest, df = df)
print('p-value:', p_value)

if p_value < alpha:
  print('Conclusion:', 'There is enough evidence to reject null hypothesis, and to say that variable accuracy_group and variable title are associated.')
else:
  print('Conclusion:', 'There is not enough evidence to reject null hypothesis, and to say that variable accuracy_group and variable title are not associated.')

del dsTrainPivot
dsTrainLabels = dsTrainLabels[['game_session', 'installation_id', 'title', 'accuracy', 'accuracy_group']]
dsTrainLabels.head()
dsTrainLabels[['accuracy_group', 'accuracy']].groupby(['accuracy_group']).agg(['mean', 'std', 'median', 'count']).reset_index()
dsTrainLabels = dsTrainLabels.drop('accuracy', axis=1)
dsTrainLabels.head()
dsTrain = pd.read_csv('../input/data-science-bowl-2019/train.csv')
print(dsTrain.shape)
dsTrain.head()
dsTrain.info()
# Get instances that have had at least one session with type assessment
keepIds = (dsTrain[dsTrain['type'] == 'Assessment']['installation_id']).drop_duplicates()

# Filter by instances
dsTrainAssessment = pd.merge(dsTrain, keepIds, how = 'inner', on = 'installation_id')
print(dsTrainAssessment.shape)
dsTrainAssessment.head()
# Filter attempts assessments by 4100 and 4110 event code
dsTrainAssessment = dsTrainAssessment[
    ((dsTrainAssessment['type'] == 'Assessment') & (dsTrainAssessment['event_code'] == 4100) & (dsTrainAssessment['title'] != 'Bird Measurer (Assessment)')) |
    ((dsTrainAssessment['type'] == 'Assessment') & (dsTrainAssessment['event_code'] == 4110) & (dsTrainAssessment['title'] == 'Bird Measurer (Assessment)'))
]
print(dsTrainAssessment.shape)
dsTrainAssessment.head()
dsTrainAssessment['event_id'].unique()
# Create is_successful variable to know which observation is correct
dsTrainAssessment['is_successful'] = dsTrainAssessment['event_data'].map(lambda x: json.loads(x)['correct'])
def getColumnStumps(x, operation = 'mean'):
  eventData = json.loads(x)
  if 'stumps' in eventData.keys():
    if operation == 'mean':
      result = np.mean(eventData['stumps'])
      result = 0 if np.isnan(result) else result 
      return result
    else:
      result = np.median(eventData['stumps'])
      result = 0 if np.isnan(result) else result 
      return result
  else:
    return np.nan
def getColumnCaterpillars(x, operation = 'mean'):
  eventData = json.loads(x)
  if 'caterpillars' in eventData.keys():
    if operation == 'mean':
      result = np.mean(eventData['caterpillars'])
      result = 0 if np.isnan(result) else result 
      return result
    else:
      result = np.median(eventData['caterpillars'])
      result = 0 if np.isnan(result) else result 
      return result
  else:
    return np.nan
def getColumnPillars(x, operation = 'mean'):
  eventData = json.loads(x)
  if 'pillars' in eventData.keys():
    if operation == 'mean':
      result = np.mean(eventData['pillars'])
      result = 0 if np.isnan(result) else result 
      return result
    else:
      result = np.median(eventData['pillars'])
      result = 0 if np.isnan(result) else result 
      return result
  else:
    return np.nan
def getColumnBuckets(x, operation = 'mean'):
  eventData = json.loads(x)
  if 'buckets' in eventData.keys():
    if operation == 'mean':
      result = np.mean(eventData['buckets'])
      result = 0 if np.isnan(result) else result 
      return result
    else:
      result = np.median(eventData['buckets'])
      result = 0 if np.isnan(result) else result 
      return result
  else:
    return np.nan
def getColumnBuckets_placed(x, operation = 'mean'):
  eventData = json.loads(x)
  if 'buckets_placed' in eventData.keys():
    if operation == 'mean':
      result = np.mean(eventData['buckets_placed'])
      result = 0 if np.isnan(result) else result 
      return result
    else:
      result = np.median(eventData['buckets_placed'])
      result = 0 if np.isnan(result) else result 
      return result
  else:
    return np.nan
def getColumnDuration(x):
  eventData = json.loads(x)
  if 'duration' in eventData.keys():
    return 0 if np.isnan(eventData['duration']) else eventData['duration']
  else:
    return np.nan
def getLeftColumnCrystals(x, operation = 'mean'):
  eventData = json.loads(x)
  if 'left' in eventData.keys():
    crystals = eventData['left']
    weights = []
    for c in crystals:
      weights.append(c['weight'])

    if operation == 'mean':
      result = np.mean(weights)
      result = 0 if np.isnan(result) else result 
      return result
    else:
      result = np.median(weights)
      result = 0 if np.isnan(result) else result 
      return result

    return result
  else:
    return np.nan
def getRightColumnCrystals(x, operation = 'mean'):
  eventData = json.loads(x)
  if 'right' in eventData.keys():
    crystals = eventData['right']
    weights = []
    for c in crystals:
      weights.append(c['weight'])

    if operation == 'mean':
      result = np.mean(weights)
      result = 0 if np.isnan(result) else result 
      return result
    else:
      result = np.median(weights)
      result = 0 if np.isnan(result) else result 
      return result

    return result
  else:
    return np.nan
dsTrainAssessment['stumps'] = dsTrainAssessment['event_data'].apply(getColumnStumps)
dsTrainAssessment['caterpillars'] = dsTrainAssessment['event_data'].apply(getColumnCaterpillars)
dsTrainAssessment['pillars'] = dsTrainAssessment['event_data'].apply(getColumnPillars)
dsTrainAssessment['buckets'] = dsTrainAssessment['event_data'].apply(getColumnBuckets)
dsTrainAssessment['buckets_placed'] = dsTrainAssessment['event_data'].apply(getColumnBuckets_placed)
dsTrainAssessment['duration'] = dsTrainAssessment['event_data'].apply(getColumnDuration)
dsTrainAssessment['left_crystals'] = dsTrainAssessment['event_data'].apply(getLeftColumnCrystals)
dsTrainAssessment['right_crystals'] = dsTrainAssessment['event_data'].apply(getRightColumnCrystals)
print(dsTrainAssessment.shape)
dsTrainAssessment.head()
del dsTrain, keepIds
dsTrainAssessment = dsTrainAssessment[['event_id', 'game_session', 'installation_id', 'game_time', 'title', 'world', 'is_successful', 
                                       'stumps', 'caterpillars', 'pillars', 'buckets', 'buckets_placed', 'duration', 'left_crystals', 'right_crystals']]
dsTrainAssessment.head()
dsTrainAssessment['world'].unique()
dsTrainAssessment[['event_id', 'world', 'game_time']].groupby(['world', 'event_id']).count().reset_index()
dsTrainAssessment[['event_id', 'title', 'game_time']].groupby(['event_id', 'title']).count().reset_index()
# Get available event_id
dsTrainAssessment['event_id'].unique()
dsTrainAssessment[dsTrainAssessment['event_id'] == '25fa8af4']['stumps'].isna().sum()
dsTrainAssessment[dsTrainAssessment['event_id'] == '17113b36']['caterpillars'].isna().sum()
dsTrainAssessment[dsTrainAssessment['event_id'] == '392e14df'][['buckets', 'buckets_placed', 'duration']].isna().sum()
dsTrainAssessment[dsTrainAssessment['event_id'] == 'd122731b'][['left_crystals', 'right_crystals']].isna().sum()
dsTrainAssessment[dsTrainAssessment['event_id'] == '93b353f2']['pillars'].isna().sum()
dsTrainAssessmentMean = dsTrainAssessment.groupby(['event_id', 'installation_id', 'world', 'game_session', 'title']).mean().reset_index()
print(dsTrainAssessmentMean.shape)
dsTrainAssessmentMean.head()
dsTrainMerge = pd.merge(dsTrainAssessmentMean, dsTrainLabels, how = 'inner', on = ['installation_id', 'game_session', 'title'])
print(dsTrainMerge.shape)
dsTrainMerge.head()
dsTrainMerge = dsTrainMerge[['world', 'title', 'game_time', 'stumps', 
                             'caterpillars', 'pillars', 'buckets', 'buckets_placed', 'duration', 'left_crystals', 'right_crystals', 'accuracy_group']]
print(dsTrainMerge.shape)
dsTrainMerge.head()
fig, (ax11, ax12) = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
ax11.hist(dsTrainMerge['game_time'], bins = 50)
ax11.set_xlabel('game_time')
ax11.set_ylabel('frecuency')
ax11.set_title('game_time distribution')

sns.boxplot(x = 'game_time', data = dsTrainMerge, ax = ax12)

plt.show()
# Calculate percentage of outliers by accuracy_group
dsTrainSummary = dsTrainMerge[['game_time', 'accuracy_group']].groupby(['accuracy_group']).count().reset_index()
dsTrainSummary['game_time_outliers'] = dsTrainMerge[dsTrainMerge['game_time'] >= 60000][['game_time', 'accuracy_group']].groupby(['accuracy_group']).count().reset_index()['game_time']
dsTrainSummary['game_time_outliers'] = round(dsTrainSummary['game_time_outliers'] / dsTrainSummary['game_time'], 2)
dsTrainSummary
# Show outliers
dsTrainMerge[dsTrainMerge['game_time'] >= 60000][['world', 'title', 'game_time', 'accuracy_group']].groupby(['world', 'title', 'accuracy_group']).count().reset_index()
dsTrainMerge = dsTrainMerge[dsTrainMerge['game_time'] < 60000]
fig, (ax11, ax12) = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
ax11.hist(dsTrainMerge['game_time'], bins = 50)
ax11.set_xlabel('game_time')
ax11.set_ylabel('frecuency')
ax11.set_title('game_time distribution')

sns.boxplot(x = 'game_time', data = dsTrainMerge, ax = ax12)

plt.show()
dsTrainMerge[dsTrainMerge['title'] == 'Bird Measurer (Assessment)'][['accuracy_group', 'caterpillars']].groupby('accuracy_group').agg(['mean', 'std']).reset_index()
fig, (ax11, ax12) = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
ax11.hist(dsTrainMerge[dsTrainMerge['title'] == 'Bird Measurer (Assessment)']['caterpillars'], bins = 20)
ax11.set_xlabel('caterpillars')
ax11.set_ylabel('frecuency')
ax11.set_title('caterpillars distribution')

sns.boxplot(x = 'accuracy_group', y = 'caterpillars', data = dsTrainMerge[dsTrainMerge['title'] == 'Bird Measurer (Assessment)'], ax = ax12)

plt.show()
dsTrainMerge[dsTrainMerge['title'] == 'Mushroom Sorter (Assessment)'][['accuracy_group', 'stumps']].groupby('accuracy_group').agg(['mean', 'std']).reset_index()
fig, (ax11, ax12) = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
ax11.hist(dsTrainMerge[dsTrainMerge['title'] == 'Mushroom Sorter (Assessment)']['stumps'], bins = 20)
ax11.set_xlabel('stumps')
ax11.set_ylabel('frecuency')
ax11.set_title('stumps distribution')

sns.boxplot(x = 'accuracy_group', y = 'stumps', data = dsTrainMerge[dsTrainMerge['title'] == 'Mushroom Sorter (Assessment)'], ax = ax12)

plt.show()
dsTrainMerge[dsTrainMerge['title'] == 'Cauldron Filler (Assessment)'][['accuracy_group', 'buckets']].groupby('accuracy_group').agg(['mean', 'std']).reset_index()
fig, (ax11, ax12) = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
ax11.hist(dsTrainMerge[dsTrainMerge['title'] == 'Cauldron Filler (Assessment)']['buckets'], bins = 20)
ax11.set_xlabel('buckets')
ax11.set_ylabel('frecuency')
ax11.set_title('buckets distribution')

sns.boxplot(x = 'accuracy_group', y = 'buckets', data = dsTrainMerge[dsTrainMerge['title'] == 'Cauldron Filler (Assessment)'], ax = ax12)

plt.show()
dsTrainMerge[dsTrainMerge['title'] == 'Cauldron Filler (Assessment)'][['accuracy_group', 'buckets_placed']].groupby('accuracy_group').agg(['mean', 'std']).reset_index()
fig, (ax11, ax12) = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
ax11.hist(dsTrainMerge[dsTrainMerge['title'] == 'Cauldron Filler (Assessment)']['buckets_placed'], bins = 20)
ax11.set_xlabel('buckets_placed')
ax11.set_ylabel('frecuency')
ax11.set_title('buckets_placed distribution')

sns.boxplot(x = 'accuracy_group', y = 'buckets_placed', data = dsTrainMerge[dsTrainMerge['title'] == 'Cauldron Filler (Assessment)'], ax = ax12)

plt.show()
dsTrainMerge[dsTrainMerge['title'] == 'Cauldron Filler (Assessment)'][['accuracy_group', 'duration']].groupby('accuracy_group').agg(['mean', 'std']).reset_index()
fig, (ax11, ax12) = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
ax11.hist(dsTrainMerge[dsTrainMerge['title'] == 'Cauldron Filler (Assessment)']['duration'], bins = 20)
ax11.set_xlabel('duration')
ax11.set_ylabel('frecuency')
ax11.set_title('duration distribution')

sns.boxplot(x = 'accuracy_group', y = 'duration', data = dsTrainMerge[dsTrainMerge['title'] == 'Cauldron Filler (Assessment)'], ax = ax12)

plt.show()
dsTrainMerge[dsTrainMerge['title'] == 'Cauldron Filler (Assessment)'][['buckets', 'buckets_placed', 'duration']].corr()
fig, (ax11, ax12) = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
sns.scatterplot(x = 'buckets', y = 'buckets_placed', data = dsTrainMerge[dsTrainMerge['title'] == 'Cauldron Filler (Assessment)'], ax = ax11)

sns.scatterplot(x = 'buckets', y = 'duration', data = dsTrainMerge[dsTrainMerge['title'] == 'Cauldron Filler (Assessment)'], ax = ax12)

plt.show()
dsTrainMerge[dsTrainMerge['title'] == 'Chest Sorter (Assessment)'][['accuracy_group', 'pillars']].groupby('accuracy_group').agg(['mean', 'std']).reset_index()
fig, (ax11, ax12) = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
ax11.hist(dsTrainMerge[dsTrainMerge['title'] == 'Chest Sorter (Assessment)']['pillars'], bins = 20)
ax11.set_xlabel('pillars')
ax11.set_ylabel('frecuency')
ax11.set_title('pillars distribution')

sns.boxplot(x = 'accuracy_group', y = 'pillars', data = dsTrainMerge[dsTrainMerge['title'] == 'Chest Sorter (Assessment)'], ax = ax12)

plt.show()
dsTrainMerge[dsTrainMerge['title'] == 'Cart Balancer (Assessment)'][['accuracy_group', 'left_crystals']].groupby('accuracy_group').agg(['mean', 'std']).reset_index()
fig, (ax11, ax12) = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
ax11.hist(dsTrainMerge[dsTrainMerge['title'] == 'Cart Balancer (Assessment)']['left_crystals'], bins = 20)
ax11.set_xlabel('left_crystals')
ax11.set_ylabel('frecuency')
ax11.set_title('left_crystals distribution')

sns.boxplot(x = 'accuracy_group', y = 'left_crystals', data = dsTrainMerge[dsTrainMerge['title'] == 'Cart Balancer (Assessment)'], ax = ax12)

plt.show()
dsTrainMerge[dsTrainMerge['title'] == 'Cart Balancer (Assessment)'][['accuracy_group', 'right_crystals']].groupby('accuracy_group').agg(['mean', 'std']).reset_index()
fig, (ax11, ax12) = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
ax11.hist(dsTrainMerge[dsTrainMerge['title'] == 'Cart Balancer (Assessment)']['right_crystals'], bins = 20)
ax11.set_xlabel('right_crystals')
ax11.set_ylabel('frecuency')
ax11.set_title('right_crystals distribution')

sns.boxplot(x = 'accuracy_group', y = 'right_crystals', data = dsTrainMerge[dsTrainMerge['title'] == 'Cart Balancer (Assessment)'], ax = ax12)

plt.show()
dsTrainMerge[dsTrainMerge['title'] == 'Cart Balancer (Assessment)'][['left_crystals', 'right_crystals']].corr()
sns.scatterplot(x = 'left_crystals', y = 'right_crystals', data = dsTrainMerge[dsTrainMerge['title'] == 'Cart Balancer (Assessment)'])
dsTrainMerge = dsTrainMerge[['world', 'title', 'game_time', 'stumps', 'caterpillars', 'pillars', 'buckets', 'duration', 'left_crystals', 'right_crystals', 'accuracy_group']]
print(dsTrainMerge.shape)
dsTrainMerge.head()
dsTest = pd.read_csv('../input/data-science-bowl-2019/test.csv')
print('Shape:', dsTest.shape)
dsTest.head()
# Get instances that have had at least one session with type assessment
keepIds = (dsTest[dsTest['type'] == 'Assessment']['installation_id']).drop_duplicates()

# Filter by instances
dsTestAssessment = pd.merge(dsTest, keepIds, how = 'inner', on = 'installation_id')
print(dsTestAssessment.shape)
dsTestAssessment.head()
# Filter attempts assessments by 4100 and 4110 event code
dsTestAssessment = dsTestAssessment[
    ((dsTestAssessment['type'] == 'Assessment') & (dsTestAssessment['event_code'] == 4100) & (dsTestAssessment['title'] != 'Bird Measurer (Assessment)')) |
    ((dsTestAssessment['type'] == 'Assessment') & (dsTestAssessment['event_code'] == 4110) & (dsTestAssessment['title'] == 'Bird Measurer (Assessment)'))
]
print(dsTestAssessment.shape)
dsTestAssessment.head()
dsTestAssessment['event_id'].unique()
dsTestAssessment = dsTestAssessment[['event_id', 'game_session', 'timestamp', 'event_data', 'installation_id', 'game_time', 'title', 'world']]
dsTestAssessment.head()
dsTestAssessment['stumps'] = dsTestAssessment['event_data'].apply(getColumnStumps)
dsTestAssessment['caterpillars'] = dsTestAssessment['event_data'].apply(getColumnCaterpillars)
dsTestAssessment['pillars'] = dsTestAssessment['event_data'].apply(getColumnPillars)
dsTestAssessment['buckets'] = dsTestAssessment['event_data'].apply(getColumnBuckets)
dsTestAssessment['buckets_placed'] = dsTestAssessment['event_data'].apply(getColumnBuckets_placed)
dsTestAssessment['duration'] = dsTestAssessment['event_data'].apply(getColumnDuration)
dsTestAssessment['left_crystals'] = dsTestAssessment['event_data'].apply(getLeftColumnCrystals)
dsTestAssessment['right_crystals'] = dsTestAssessment['event_data'].apply(getRightColumnCrystals)
dsTestAssessment['is_successful'] = dsTestAssessment['event_data'].map(lambda x: json.loads(x)['correct'])
print(dsTestAssessment.shape)
dsTestAssessment.head()
dsTestAssessment = dsTestAssessment[['installation_id', 'timestamp', 'game_session', 'world', 'title', 'game_time', 'stumps', 'caterpillars', 'pillars', 'buckets', 'duration', 'left_crystals', 'right_crystals', 'is_successful']]
dsTestAssessment.head()
# Get last assessment by installation_id
dsTestAssessmentLast = dsTestAssessment[['installation_id', 'timestamp']].groupby('installation_id').max().reset_index()
print(dsTestAssessmentLast.shape)
dsTestAssessmentLast.head()
# Filter by instances
dsTestAssessment = pd.merge(dsTestAssessment, dsTestAssessmentLast, how = 'inner', on = ['installation_id', 'timestamp'])
print(dsTestAssessment.shape)
dsTestAssessment.head()
dsTestAssessment = dsTestAssessment[['installation_id', 'world', 'title', 'game_time', 'stumps', 'caterpillars', 'pillars', 'buckets', 'duration', 'left_crystals', 'right_crystals', 'is_successful']]
dsTestAssessment.head()
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
x = dsTrainMerge.drop('accuracy_group', axis = 1)
y = dsTrainMerge['accuracy_group']

print('x shape:', x.shape)
print('y shape:', y.shape)
x['game_time'] =  MinMaxScaler().fit_transform(x['game_time'].values.reshape(-1, 1))
x['stumps'] =  MinMaxScaler().fit_transform(x['stumps'].values.reshape(-1, 1))
x['caterpillars'] =  MinMaxScaler().fit_transform(x['caterpillars'].values.reshape(-1, 1))
x['pillars'] =  MinMaxScaler().fit_transform(x['pillars'].values.reshape(-1, 1))
x['buckets'] =  MinMaxScaler().fit_transform(x['buckets'].values.reshape(-1, 1))
x['duration'] =  MinMaxScaler().fit_transform(x['duration'].values.reshape(-1, 1))
x['left_crystals'] =  MinMaxScaler().fit_transform(x['left_crystals'].values.reshape(-1, 1))
x['right_crystals'] =  MinMaxScaler().fit_transform(x['right_crystals'].values.reshape(-1, 1))
x.head()
x = x.fillna(0)
x.head()
x = pd.concat([pd.get_dummies(x[['world', 'title']], drop_first = True), x[['game_time', 'stumps', 'caterpillars', 'pillars', 
                                                                                    'buckets', 'duration', 'left_crystals', 'right_crystals']]], axis = 1)
print(x.shape)
x.head()
x_train, x_val, y_train, y_val = train_test_split(x.values, y.values, test_size=0.1, random_state=0)

print('x_train shape', x_train.shape)
print('x_val shape', x_val.shape)
print('y_train shape', y_train.shape)
print('y_val shape', y_val.shape)
dsTestAssessment['game_time'] =  MinMaxScaler().fit_transform(dsTestAssessment['game_time'].values.reshape(-1, 1))
dsTestAssessment['stumps'] =  MinMaxScaler().fit_transform(dsTestAssessment['stumps'].values.reshape(-1, 1))
dsTestAssessment['caterpillars'] =  MinMaxScaler().fit_transform(dsTestAssessment['caterpillars'].values.reshape(-1, 1))
dsTestAssessment['pillars'] =  MinMaxScaler().fit_transform(dsTestAssessment['pillars'].values.reshape(-1, 1))
dsTestAssessment['buckets'] =  MinMaxScaler().fit_transform(dsTestAssessment['buckets'].values.reshape(-1, 1))
dsTestAssessment['duration'] =  MinMaxScaler().fit_transform(dsTestAssessment['duration'].values.reshape(-1, 1))
dsTestAssessment['left_crystals'] =  MinMaxScaler().fit_transform(dsTestAssessment['left_crystals'].values.reshape(-1, 1))
dsTestAssessment['right_crystals'] =  MinMaxScaler().fit_transform(dsTestAssessment['right_crystals'].values.reshape(-1, 1))
dsTestAssessment.head()
dsTestAssessment = dsTestAssessment.fillna(0)
dsTestAssessment.head()
dsTestAssessment = pd.concat([pd.get_dummies(dsTestAssessment[['world', 'title']], drop_first = True), dsTestAssessment[['installation_id', 'game_time', 'stumps', 'caterpillars', 'pillars', 
                                                                                    'buckets', 'duration', 'left_crystals', 'right_crystals']]], axis = 1)
print(dsTestAssessment.shape)
dsTestAssessment.head()
from sklearn.model_selection import validation_curve
from sklearn import metrics

from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn import tree
from sklearn import svm

from sklearn.model_selection import RandomizedSearchCV
x_val.shape
clf = MultinomialNB()
clf.fit(x_train, y_train)
y_pred = clf.predict(x_val)
acc = metrics.accuracy_score(y_val, y_pred)
cfm = metrics.confusion_matrix(y_val, y_pred)
f1 = metrics.f1_score(y_val, y_pred, average = 'weighted')

print('Accuracy:', acc)
print('F1-score:', f1)
cfm
modelTree = tree.DecisionTreeClassifier()
modelTree.fit(x_train, y_train)
y_pred = modelTree.predict(x_val)
acc = metrics.accuracy_score(y_val, y_pred)
cfm = metrics.confusion_matrix(y_val, y_pred)
f1 = metrics.f1_score(y_val, y_pred, average = 'weighted')

print('Accuracy:', acc)
print('F1-score:', f1)
# Confusion matrix
cfm
svc = svm.SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_val)
acc = metrics.accuracy_score(y_val, y_pred)
cfm = metrics.confusion_matrix(y_val, y_pred)
f1 = metrics.f1_score(y_val, y_pred, average = 'weighted')

print('Accuracy:', acc)
print('F1-score:', f1)
cfm
linSvc = svm.LinearSVC()
linSvc.fit(x_train, y_train)
y_pred = linSvc.predict(x_val)
acc = metrics.accuracy_score(y_val, y_pred)
cfm = metrics.confusion_matrix(y_val, y_pred)
f1 = metrics.f1_score(y_val, y_pred, average = 'weighted')

print('Accuracy:', acc)
print('F1-score:', f1)
cfm
x_test = dsTestAssessment.drop('installation_id', axis=1).values
print('Shape:', x_test.shape)
y_predTest = modelTree.predict(x_test)
y_predTest
dsSubmission = pd.DataFrame({'installation_id': dsTestAssessment['installation_id'].values, 'accuracy_group': y_predTest})
dsSubmission.head()
dsSubmission.to_csv('submission.csv', index=False)
