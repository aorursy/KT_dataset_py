# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import seaborn as sns

import matplotlib.pyplot as plt



from numba import njit

from itertools import product

from ortools.linear_solver import pywraplp

%matplotlib inline
df = pd.read_csv('/kaggle/input/santa-workshop-tour-2019/family_data.csv', index_col = 'family_id')

submission_df = pd.read_csv('/kaggle/input/santa-workshop-tour-2019/sample_submission.csv', index_col = 'family_id')

prediction = submission_df.assigned_day.values
df.head(10)
df.shape
df.describe()
submission_df.shape
submission_df.head(10)
# Choice columns list

choice_cols = list(df.columns)

choice_cols.remove('n_people')

choice_dict = df[choice_cols].to_dict()
# Distribution plot for choice columns

fig, axes = plt.subplots(5, 2, figsize = (16,12))

axes = axes.ravel()

for i, choice in enumerate(choice_cols):

    ax = axes[i]

    sns.distplot(df.loc[:, choice], ax = ax, label = choice, color = 'blue')

plt.tight_layout()
# Family size Distribution

family_size = df['n_people'].value_counts().sort_index()

plt.figure(figsize = (17,8))

ax = sns.barplot(x = family_size.index, y = family_size.values)

for p in ax.patches:

    ax.annotate(f'{p.get_height():.0f}', xy = (p.get_x() + p.get_width()/ 2., p.get_height()), xytext = (-10, 5), textcoords = 'offset points')

plt.xlabel('Family Size', fontsize = 14)

plt.ylabel('No of families', fontsize = 14)

plt.title('Family Members Distribution', fontsize = 14)

plt.show()
N_DAYS = 100

MAX_OCCUPANCY = 300

MIN_OCCUPANCY = 125



# In reverse order from 100 to 0

days = list(range(N_DAYS,0,-1))
# Wednesday is 0th day since 25th Dec 2019 is Wednesday

def dayofweek(days):

    dayofweek = []

    for day in days:

        if day%7 == 2:

            dayofweek.append('Monday')

        elif day%7 == 1:

            dayofweek.append('Tuesday')

        elif day%7 == 0:

            dayofweek.append('Wednesday')

        elif day%7 == 6:

            dayofweek.append('Thursday')

        elif day%7 == 5:

            dayofweek.append('Friday')

        elif day%7 == 4:

            dayofweek.append('Saturday')

        else:

            dayofweek.append('Sunday')

    return dayofweek
dayofweeklist = [dayofweek(df[choice_cols].values[i][:].tolist()) for i in range(5000)]

dayofweek_df = pd.DataFrame(dayofweeklist, columns = ['dayofweek' + choice_cols[i] for i in range(10)])

dayofweek_df
plt.figure(figsize = (17,8))

sns.countplot(dayofweek_df['dayofweekchoice_0'], order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

plt.xlabel('Choice 0', fontsize = 14)

plt.ylabel('Count', fontsize = 14)

plt.title('Weekday concentration for Choice 0', fontsize = 14)

plt.show()
plt.figure(figsize = (17,8))

sns.countplot(dayofweek_df['dayofweekchoice_9'], order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

plt.xlabel('Choice 9', fontsize = 14)

plt.ylabel('Count', fontsize = 14)

plt.title('Weekday concentration for Choice 9', fontsize = 14)

plt.show()
def penalty_for_cost(family_members):

    choice_penalty = {}

    choice_penalty[1] = 50

    choice_penalty[2] = 50 + 9 * family_members

    choice_penalty[3] = 100 + 9 * family_members

    choice_penalty[4] = 200 + 9 * family_members

    choice_penalty[5] = 200 + 18 * family_members

    choice_penalty[6] = 300 + 18 * family_members

    choice_penalty[7] = 400 + 36 * family_members

    choice_penalty[8] = 500 + (36 + 199) * family_members

    choice_penalty[9] = 500 + (36 + 398) * family_members

    

    items = choice_penalty.items()

    return list(zip(*items))
plt.figure(figsize = (17, 8))

for i in range(2,9,1):

    indices, cost = penalty_for_cost(i)

    plt.plot(indices, cost, label = f'{i} family members')

plt.xlabel('Choice', fontsize = 14)

plt.ylabel('Cost', fontsize = 14)

plt.title('Choice vs Cost plot for all family sizes', fontsize = 14)

plt.legend()

plt.show()
penalties = np.asarray([

    [

        0,

        50,

        50 + 9 * n,

        100 + 9 * n,

        200 + 9 * n,

        200 + 18 * n,

        300 + 18 * n,

        300 + 36 * n,

        400 + 36 * n,

        500 + 36 * n + 199 * n,

        500 + 36 * n + 398 * n

    ] for n in range(family_size.max() + 1)

])



family_cost_matrix = np.concatenate(df.n_people.apply(lambda n: np.repeat(penalties[n, 10], 100).reshape(1, 100)))

for family in df.index:

    for choice_order, day in enumerate(df.loc[family].drop('n_people')):

        family_cost_matrix[family, day - 1] = penalties[df.loc[family, 'n_people'], choice_order]
accounting_cost_matrix = np.zeros((500, 500))

for n in range(accounting_cost_matrix.shape[0]):

    for diff in range(accounting_cost_matrix.shape[1]):

        accounting_cost_matrix[n, diff] = max(0, (n - 125.0) / 400.0 * n ** (0.5 + diff / 50.0))
@njit(fastmath = True)

def cost_function(prediction, family_size, family_cost_matrix, accounting_cost_matrix):

    N_DAYS = family_cost_matrix.shape[1]

    MAX_OCCUPANCY = 300

    MIN_OCCUPANCY = 125

    penalty = 0

    daily_occupancy = np.zeros(N_DAYS + 1, dtype = np.int16)

    for i, (pred, n) in enumerate(zip(prediction, family_size)):

        daily_occupancy[pred - 1] += n

        penalty += family_cost_matrix[i, pred - 1]

    accounting_cost = 0

    n_high = 0

    n_low = 0

    daily_occupancy[-1] = daily_occupancy[-2]

    for day in range(N_DAYS):

        n_next = daily_occupancy[day + 1]

        n = daily_occupancy[day]

        n_high += (n > MAX_OCCUPANCY)

        n_low += (n < MIN_OCCUPANCY)

        diff = abs(n - n_next)

        accounting_cost += accounting_cost_matrix[n, diff]

    return np.asarray([penalty, accounting_cost, n_high, n_low])
family_size = df.n_people.values.astype(np.int16)
start_penalty, accounting_cost, high, low = cost_function(prediction, family_size, family_cost_matrix, accounting_cost_matrix)

start_cost = start_penalty + accounting_cost
new = list(prediction)

for family_id, _ in enumerate(new):

    for choice_pick in range(10):

        day = choice_dict[f'choice_{choice_pick}'][family_id]

        temp = new.copy()

        temp[family_id] = day

        temp = np.asarray(temp)

        cur_penalty, cur_accounting_cost ,h,l = cost_function(temp, family_size, family_cost_matrix, accounting_cost_matrix)

        cur_cost = cur_penalty + cur_accounting_cost

        if cur_cost < start_cost:

            new = temp.copy()

            start_cost = cur_cost
submission_df['assigned_day'] = new

score = start_cost

submission_df.to_csv(f'submission_{score}.csv')

print(f'Score : {score}')