import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from pandas.plotting import register_matplotlib_converters



scatter_figsize = (11, 4)

hist_figsize = (8, 2)



cmap = plt.get_cmap('tab10')

add_color = np.array([[0., 0.56, 0.45, 1.]])

my_colors = np.append(cmap(range(10)), add_color, axis = 0)

base = os.path.abspath('/kaggle/input/data-without-drift/')



_x = [0, 100, 150, 200, 250, 300, 350, 400, 450, 500]

train_borders = [(_x[i], _x[i + 1]) for i in range(len(_x) - 1)]

_y = [500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 650, 700]
def scatter_colored_by_target(df, col_name='signal', show=True, 

                              colors=my_colors, title=None):

    

    plt.rcParams['figure.figsize'] = scatter_figsize

    

    target_array = np.sort(df.open_channels.unique())

    for target_value in target_array:

        color = colors[target_value]

        _df = df[df.open_channels == target_value]

        plt.scatter(_df.time, _df[col_name], c=color, label=target_value)

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

    
train = pd.read_csv(os.path.join(base + '/train_clean.csv'))

test = pd.read_csv(os.path.join(base + '/test_clean.csv'))
scatter_colored_by_target(train, show=False)

for x in _x:

    plt.axvline(x, c='k')



plt.xticks(_x)

plt.show()
plt.rcParams['figure.figsize'] = scatter_figsize

plt.scatter(test.time, test.signal)



for x in _y:

    plt.axvline(x, c='k')



plt.xticks(np.arange(500, 700, 10))

plt.show()
for num, (left, right) in enumerate(train_borders):

    temp_df = train[(train.time > left) & (train.time <= right)]

    stats_df = temp_df.groupby(['open_channels'])['signal'].agg(

        ['mean', 'std', 'min', 'max', 'count'])

    stats_df['percentage'] = stats_df['count']/ len(temp_df)

    

    title = f'Case {num}. For time in [{left}, {right}).' 

    

    print(title)

    print(stats_df)

    scatter_colored_by_target(temp_df, title=title)

    hist_colored_by_target(temp_df, title=title)