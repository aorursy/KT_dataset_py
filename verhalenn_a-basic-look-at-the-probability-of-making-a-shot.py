import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



data = pd.read_csv('../input/shot_logs.csv')
print('Data Shape:', data.shape)

print('Data Columns:\n', data.columns)
data['SHOT_RESULT'].value_counts(normalize=True)
data['DRIBBLES'].value_counts()
dribbles_results = data[data['DRIBBLES'] < 13].groupby('DRIBBLES')['SHOT_RESULT'].value_counts(normalize=True)
dribbles_results.loc[:, 'made'].plot()

plt.ylabel('Chance of Success')

plt.show()
data[data['DRIBBLES'] >= 13]['SHOT_RESULT'].value_counts(normalize=True)
# Calculate shot clock percentages.

time_data = pd.crosstab(data['SHOT_CLOCK'], data['SHOT_RESULT'], normalize='index').loc[:, 'made']

time_data.plot()

# Create a base line.

plt.plot([max(time_data.index), min(time_data.index)], [.45, .45])

# Label and show the plot.

plt.ylabel('Chance of Making the Shot')

plt.xlabel('Shot Clock')

plt.show()
# Calculate the number of dribbels by shot clock time.

data.groupby('SHOT_CLOCK')['DRIBBLES'].mean().plot()

# Label and show the plot.

plt.xlabel('Shot Clock')

plt.ylabel('Average Number of Dribbles')

plt.show()
import seaborn as sns



sns.distplot(data['CLOSE_DEF_DIST'])

plt.show()
# Plot the density of shots made and shots missed in comparison with the closest defender.

sns.distplot(data[data['SHOT_RESULT'] == 'made']['CLOSE_DEF_DIST'], label='made')

sns.distplot(data[data['SHOT_RESULT'] == 'missed']['CLOSE_DEF_DIST'], label='missed')

# Label and show the plot.

plt.xlabel('Distance to the Closest Defender')

plt.ylabel('Density')

plt.legend()

plt.show()
closest_defender_chance = pd.crosstab(data['CLOSE_DEF_DIST'], data['SHOT_RESULT'], normalize='index').loc[:,'made']

closest_defender_chance.plot()

plt.plot([max(closest_defender_chance.index), min(closest_defender_chance.index)], [.45, .45])

plt.xlabel('Distance to the Closest Defender')

plt.ylabel('Chance of Making a Shot')

plt.show()
shot_dist_chance = pd.crosstab(data['SHOT_DIST'], data['SHOT_RESULT'], normalize='index').loc[:, 'made']

shot_dist_chance.plot()

plt.plot([max(shot_dist_chance.index), min(shot_dist_chance.index)], [.45, .45])

plt.xlabel('Shot Distance')

plt.ylabel('Chance of Making the Shot')

plt.show()
data['PERIOD'].value_counts()
regulation_data = data[data['PERIOD'] <= 4]

regulation_data.shape
pd.crosstab(regulation_data['PERIOD'], regulation_data['SHOT_RESULT'], normalize='index').loc[:, 'made']
data['MINUTE'] = data['GAME_CLOCK'].str.split(':').str.get(0).astype(int)

for period in range(1, 5):

    period_data = data[data['PERIOD'] == period]

    pd.crosstab(period_data['MINUTE'], period_data['SHOT_RESULT'], normalize='index').loc[:, 'made'].plot(label=period)

    

plt.xlabel('Minutes Left in the Period')

plt.ylabel('Chance of Making the Shot')

plt.legend()

plt.show()
