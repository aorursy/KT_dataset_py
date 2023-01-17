import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from math import sqrt, log, inf

from sklearn.metrics import f1_score, accuracy_score



import warnings

warnings.filterwarnings("ignore")
# computes the boundary between two groups using their means and stds using quadratic equation.

def bounder(m1, s1, m2, s2):

    if s1 == s2:

        return (m1 + m2) / 2



    a = s2 ** 2 - s1 ** 2

    b = m1 * s2 ** 2 - m2 * s1 ** 2

    c = (m1 * s2) ** 2 - (m2 * s1) ** 2 - s1 ** 2 * s1 ** 2 * log(s1 / s2)



    d = b ** 2 - a * c



    if d < 0:

        print("d < 0!")

        return (m1 + m2) / 2



    x = b + sqrt(d)

    x /= a



    if not m1 < x < m2:

        x = b - sqrt(d)

        x /= a



    return x



class Sampler:

    train = None

    test = None

    _x = [0, 50, 60, 100, 150, 200, 250, 300, 350, 400, 450, 500]

    _y = [500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 650, 700]

    train_borders = None

    test_borders = None

    case_mapping = {0: 0, 1: 4, 2: 6, 3: 0, 4: 3, 5: 5, 6: 6, 7: 5, 8: 0, 9: 4, 10: 0, 11: 0}

    train_cases = None

    open_channels_sample = None

    stats = None

    sample_cases = None

    

    def __init__(self, train, test=None, case_mapping=None):

        self.train = train

        if test is None:

            self.test = train

            self._y = self._x

            self.case_mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 3, 8: 4, 9: 6, 10: 5}

        else:

            self.test = test



        if case_mapping is not None:

            self.case_mapping = case_mapping



        self.refresh_borders()

        self.train['open_channels'] = self.train['open_channels'].astype('int64')



    def refresh_borders(self):

        self.train_borders = [(self._x[i], self._x[i + 1]) for i in range(len(self._x) - 1)]

        self.test_borders = [(self._y[i], self._y[i + 1]) for i in range(len(self._y) - 1)]

    

    def collect_stats(self):

        self.stats = {}

        # stats structure: { case index -> {target value -> {'mean': x, 'std': y, 'count': z} } }

        for num, (left, right) in enumerate(self.train_borders):

            self.stats[num] = {}

            temp_df = self.train[(self.train.time > left) & (self.train.time <= right)]

            grp_df = temp_df.groupby(['open_channels'])['signal'].agg(['mean', 'std'])

            for indx in grp_df.index:

                self.stats[num][indx] = {'mean': grp_df.loc[indx, 'mean'],

                                         'std': grp_df.loc[indx, 'std']}

    

    # computes еру array of borders for each case of data.

    def compute_bounds(self):

        self.collect_stats()



        self.train_cases = {}

        for case_num in self.stats:



            raw_case = self.stats[case_num]



            target_values = [target_value for target_value in raw_case]

            target_values.sort()

            case = [(raw_case[target_value]['mean'],

                     raw_case[target_value]['std']) for target_value in target_values]



            bounds = []

            for params, next_params in zip(case[:-1], case[1:]):

                bound = bounder(*params, *next_params)

                bounds.append(bound)



            bounds = [-inf] + bounds + [inf]



            self.train_cases[case_num] = {'bounds': bounds, 'values': target_values}



    # Computes prediction for test data using existing borders of train data and case_mapping.

    def compute_samples(self):

        result = []

        self.sample_cases = {}

        for num, (left, right) in enumerate(self.test_borders):

            temp_df = self.test[(self.test.time > left) & (self.test.time <= right)]

            signal = list(temp_df.signal)

            predicted_part = -np.ones(len(temp_df))



            case = self.train_cases[self.case_mapping[num]]

            bounds = case['bounds']

            target = case['values']



            for i, signal_value in enumerate(signal):

                for target_value, bound, next_bound in zip(target, bounds[:-1], bounds[1:]):

                    if bound <= signal_value <= next_bound:

                        predicted_part[i] = target_value



            assert -1 not in predicted_part



            result.extend(predicted_part)

            self.sample_cases[num] = np.array(predicted_part)



        self.open_channels_sample = np.array(result)



        return self.open_channels_sample
scatter_figsize = (11, 4)

hist_figsize = (10, 3)

markersize = 1

cmap = plt.get_cmap('tab10')

add_color = np.array([[0., 0.56, 0.45, 1.]])

colors = [[color] for color in cmap(range(10))]

colors.append(add_color)

my_colors = np.array(colors)





def scatter_colored_by_target(df, col_name='signal', show=True,

                              colors=my_colors, title=None):

    plt.rcParams['figure.figsize'] = scatter_figsize



    target_array = np.sort(df.open_channels.unique())

    for target_value in target_array:

        color = colors[target_value]

        _df = df[df.open_channels == target_value]

        plt.scatter(_df.time, _df[col_name], c=color, label=target_value, s=markersize)

    if title is not None:

        plt.title(title)

    plt.xlabel('time')

    plt.ylabel('signal')

    plt.legend(title='open channels', loc='upper right',

               bbox_to_anchor=(1.15, 1))

    if show:

        plt.show()





def hist_colored_by_target(df, col_name='signal', show=True,

                           colors=my_colors, title=None):

    plt.rcParams['figure.figsize'] = hist_figsize



    target_array = np.sort(df.open_channels.unique())

    for target_value in target_array:

        color = colors[target_value]

        plt.hist(df[df.open_channels == target_value][col_name],

                 color=color, label=target_value)



    plt.legend(title='open channels', loc='upper right',

               bbox_to_anchor=(1.2, 1))

    if title is not None:

        plt.title(title)

    if show:

        plt.show()

base = os.path.abspath('/kaggle/input/clean-kalman/clean_kalman/')

train = pd.read_csv(os.path.join(base + '/train_clean_kalman.csv'))

test = pd.read_csv(os.path.join(base + '/test_clean_kalman.csv'))
train_model = Sampler(train)

train_model.compute_bounds()

train_y = train_model.compute_samples()
accuracy_score(train.open_channels, train_y)
f1_score(train.open_channels, train_y, average = 'macro')
train['y'] = train_y



for num, (left, right) in enumerate(train_model.train_borders):

    temp_df = train[(train.time > left) & (train.time <= right)]

    temp_y = temp_df.y

    

    local_accuracy = accuracy_score(temp_df.open_channels, temp_y)

    local_f1 = f1_score(temp_df.open_channels, temp_y, average = 'macro')

    

    print(f''' On case {num}: accuracy={local_accuracy}, f1_score={local_f1}''')
case_num = 4

left, right = train_model.train_borders[case_num]



temp_df = train[(train.time > left) & (train.time <= right)]



title = f'Case {case_num}. For time in [{left}, {right}).' 

scatter_colored_by_target(temp_df, title=title, show=False)

for x in train_model.train_cases[case_num]['bounds'][1:-1]:

    plt.axhline(x, c='k', linewidth=3)

plt.show()



hist_colored_by_target(temp_df, title=title, show=False)

for x in train_model.train_cases[case_num]['bounds'][1:-1]:

    plt.axvline(x, c='k', linewidth=3)

plt.show()
model = Sampler(train, test)

model.compute_bounds()

y = model.compute_samples()



sample_sub = pd.read_csv('/kaggle/input/liverpool-ion-switching/sample_submission.csv')

sample_sub['open_channels'] = np.array(y).astype('int64')

sample_sub.to_csv('submission_0.csv', index=False, float_format='%.4f')
from sklearn.ensemble import RandomForestClassifier

def fe(_df, sampler, y, name='y'):

    df = _df.copy()

    

    for num, (left, right) in enumerate(sampler.test_borders):

        temp_df = df[(df.time > left) & (df.time <= right)]

        temp_df['batch'] = num

        df.update(temp_df['batch'] )

    

    df[name] = y

    df[f'prev_{name}'] = [0] + list(df[name][:-1])

    df[f'next_{name}'] = list(df[name][1:]) + [0]

    df[f'{name}_neighborhood_equal']  = (df[f'prev_{name}'] == df[f'next_{name}']).astype('int64')

    

    return df



new_train = fe(train, train_model, train_y)

cols = ['signal', 'y', 'prev_y', 'next_y', 'y_neighborhood_equal']

X = new_train[cols].values

open_channels = new_train.open_channels



new_test = fe(test, model, y)

test_X = new_test[cols].values





new_train.head()
# RF hyperparameters taken from this notebook https://www.kaggle.com/sggpls/shifted-rfc-pipeline

clf = RandomForestClassifier(n_estimators=150, max_depth=19,  

                             random_state=42, n_jobs=10, verbose=2)



clf.fit(X, open_channels)
pred_train_y = clf.predict(X)

accuracy = accuracy_score(train.open_channels, pred_train_y)

f1 = f1_score(train.open_channels, pred_train_y, average = 'macro')

print(f'accuracy={accuracy}, f1_score={f1}')
pred_y = clf.predict(test_X)

sample_sub = pd.read_csv('/kaggle/input/liverpool-ion-switching/sample_submission.csv')

sample_sub['open_channels'] = np.array(pred_y).astype('int64')

sample_sub.to_csv('submission.csv', index=False, float_format='%.4f')
f_imp = pd.DataFrame({'feature': cols, 

                      'importance': clf.feature_importances_}

                    ).sort_values('importance', ascending = False).reset_index()

f_imp['importance_normalized'] = f_imp['importance'] / f_imp['importance'].sum()

plt.figure()

ax = plt.subplot()

ax.barh(list(reversed(list(f_imp.index[:15]))), 

        f_imp['importance_normalized'].head(15), 

        align = 'center', edgecolor = 'k')

ax.set_yticks(list(reversed(list(f_imp.index[:15]))))

ax.set_yticklabels(f_imp['feature'].head(15))

plt.show()