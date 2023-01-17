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



import numpy as np

import matplotlib.pyplot as plt

from IPython.core.pylabtools import figsize

import seaborn as sns

import pandas as pd

from tqdm import tqdm
TRAIN_PATH = '/kaggle/input/conways-reverse-game-of-life-2020/train.csv'

ONE_STEP_START_PATH = '/kaggle/working/onestepstart.npy'

ONE_STEP_STOP_PATH = '/kaggle/working/onestepstop.npy'

trainset = pd.read_csv(TRAIN_PATH)
def gen_grid(N = 20, live_pct = 0.3):

    '''Generate a grid with size N * N and dencity = live_pct'''

    return (np.random.random((N,N)) > (1 - live_pct)).astype(int)





def n_lives(g):

    '''part of grid g * fileter f element size, sum to get number of live neighbors

     for each cell

    '''

    f = np.array([

          [1., 1., 1.],

          [1., 0., 1.], 

          [1., 1., 1.]])

    return (g * f).sum()





def pad_grid(grid):

    # padd with zeros around grid

    padded_grid = np.pad(grid,((1,1),(1,1)))



    # for each edge, pad edge of opposite side

    padded_grid[0,1:-1] = padded_grid[-2,1:-1]

    padded_grid[-1,1:-1] = padded_grid[1,1:-1]

    padded_grid[1:-1,0] = padded_grid[1:-1,-2]

    padded_grid[1:-1,-1] = padded_grid[1:-1,1]



    # for each coner, pad with opposite corner

    padded_grid[0,0] = grid[-1,-1]

    padded_grid[-1,-1] = grid[0,0]

    padded_grid[0,-1] = grid[-1,0]

    padded_grid[-1,0] = grid[0,-1]

    return padded_grid





def neighbor_count_grid(N):

    '''count number of neighbors for each cell'''

    neighbor_count = 8 * np.ones((N,N))

    neighbor_count[[0, N-1],:] = 5

    neighbor_count[:,[0, N-1]] = 5

    neighbor_count[0,0] = neighbor_count[0,N-1] = 3

    neighbor_count[N-1,0] = neighbor_count[N-1,N-1] = 3

    return neighbor_count





def board_lives(grid):

    '''for all cell in grid, count live neighbors'''

    n = grid.shape[0]

    padded_grid = pad_grid(grid)

    lives = np.zeros((n,n))

    for i in range(1, n + 1):

        for j in range(1, n + 1):

            lives[i-1,j-1] = n_lives(padded_grid[i-1: i+2, j-1:j+2])

    return lives





def board_dies(lives, neighbors):

    '''for all cells count died neigbors'''

    return neighbors - lives





def evolve(grid, lives):



    '''evolve base'''

    new_grid = grid.copy()



    # criterion

    over_population = np.where(grid * lives > 3)

    stasis = np.where(np.logical_or(grid * lives == 2, grid * lives == 3))

    under_population = np.where(grid * lives < 2)

    reproduction = np.where(np.logical_not(grid).astype(int) * lives == 3)



    # apply criterion

    new_grid[over_population] = 0

    new_grid[stasis] = 1

    new_grid[under_population] = 0

    new_grid[reproduction] = 1

    return new_grid



def one_step_evolve(grid):

    '''evolve one step, including live counts and rule apply'''

    lives = board_lives(grid)

    evolved_grid = evolve(grid, lives)

    return evolved_grid



def n_step_evolve(grid, steps):

    '''evolve many steps, 55-70 iterations per second'''

    for step in range(steps):

        grid = one_step_evolve(grid)

    return grid





def show_all(grid, padded_grid, lives, dies):

    '''generate visualizations'''

    figsize(15,12)

    plt.subplot(2,2,1)

    sns.heatmap(grid, linewidth = 1)

    plt.title('Grid State')

    plt.subplot(2,2,2)

    sns.heatmap(padded_grid, linewidth = 1)

    plt.title('Padded State')

    plt.subplot(2,2,3)

    sns.heatmap(lives, annot = True, linewidth = 1)

    plt.title('Live State')

    plt.subplot(2,2,4)

    sns.heatmap(dies, annot = True, linewidth = 1)

    plt.title('Die State')

    plt.show()
N = 20

live_pct = 0.3

grid = gen_grid(N, live_pct)

padded_grid = pad_grid(grid)

neighbor_count=  neighbor_count_grid(N)

lives = board_lives(grid)

dies = board_dies(lives, neighbor_count)

show_all(grid, padded_grid, lives, dies)

evolved_grid = evolve(grid, lives)





figsize(15,5)

plt.subplot(1,2,1)

sns.heatmap(grid, linewidth = 1)

plt.subplot(1,2,2)

sns.heatmap(evolved_grid,linewidth = 1)

plt.show()
grids = trainset.copy()
start_grids = grids[[col for col in grids.columns if col.startswith('start')]]

stop_grids = grids[[col for col in grids.columns if col.startswith('stop')]]

grid_steps = grids.delta.values



N = 25



accs = []

for idx in tqdm(range(len(start_grids))[:1000]):

    start = start_grids.iloc[idx].values.reshape(25,25)

    stop = stop_grids.iloc[idx].values.reshape(25,25)

    steps = grid_steps[idx]

    gen_stop = n_step_evolve(start, steps)

    accuracy = (stop == gen_stop).mean()

    accs.append(accuracy)



'Correct Code' if np.mean(accs) == 1 else 'Bugs exists'
N = 25

idx =  np.random.choice(np.arange(len(start_grids)))

start = start_grids.iloc[idx].values.reshape(25,25)

stop = stop_grids.iloc[idx].values.reshape(25,25)

steps = grid_steps[idx]

gen_stop = n_step_evolve(start, steps)

accuracy = (stop == gen_stop).mean()

print('Generation accuracy: {}'.format(accuracy))





figsize(28,6)

plt.subplot(1,4,1)

sns.heatmap(start,  linewidth = 1)

plt.title('Start')



plt.subplot(1,4,2)

sns.heatmap(stop, linewidth = 1)

plt.title('Result')





plt.subplot(1,4,3)

sns.heatmap(gen_stop, linewidth = 1)

plt.title('Generate')



plt.subplot(1,4,4)

sns.heatmap(gen_stop == stop, linewidth = 1)

plt.title('Error location')

plt.show()
# def generate_onestep_samples(start_grids, grid_steps):

#     '''Use given start grids, grid steps to generate one step samles

#      Use implemented algorithm to generate next step

#      Each start grid generate a sequence of start and stop samples (all onestep)

#     '''



#     start_samples_onestep = []

#     stop_samples_onestep = []



#     for idx in tqdm(range(len(start_grids))):

#         sta = start_grids.iloc[idx].values.reshape(25,25)

#         for st in range(grid_steps[idx]):

#             sto = n_step_evolve(sta,1)

#             start_samples_onestep.append(sta.reshape(625))

#             stop_samples_onestep.append(sto.reshape(625))

#             sta = sto

        

#     np.save(ONE_STEP_START_PATH, np.array(start_samples_onestep))

#     np.save(ONE_STEP_STOP_PATH, np.array(stop_samples_onestep))

#     return np.array(start_samples_onestep),np.array(stop_samples_onestep)
# start_samples_onestep, stop_samples_onestep = generate_onestep_samples(start_grids, grid_steps)
# figsize(16,6)

# plt.subplot(1,2,1)



# idx = np.random.randint(0, len(start_samples_onestep))

# sns.heatmap(start_samples_onestep[idx].reshape(25,25), linewidth = 1)

# plt.title('Start Sample')

# plt.subplot(1,2,2)

# sns.heatmap(stop_samples_onestep[idx].reshape(25,25), linewidth = 1)

# plt.title('Stop Sample')





# (start_samples_onestep[idx].reshape(25,25) == stop_samples_onestep[idx].reshape(25,25)).mean()
# results = [[start_samples_onestep[i].mean(), (start_samples_onestep[i] == stop_samples_onestep[i]).mean()] for i in range(len(start_samples_onestep))]

# figsize(8,6)

# results = np.array(results)

# plt.scatter(results[:,0], results[:,1], alpha = 0.1)

# plt.xlabel('Board density')

# plt.ylabel('Percent stop equal start')

# plt.show()
# plt.hist(start_samples_onestep.mean(axis = 1))

# plt.show()
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Input, Flatten,GlobalMaxPooling2D, GlobalAveragePooling2D, Concatenate, BatchNormalization

from tensorflow.keras.layers import Embedding, Reshape, Dot, Multiply

from tensorflow.keras.layers import Activation

from tensorflow.keras.models import Model, Sequential

from tensorflow.keras.utils import plot_model

from tensorflow.keras.optimizers import Adam, SGD, RMSprop

import tensorflow as tf
start_grids = grids[[col for col in grids.columns if col.startswith('start')]]

stop_grids = grids[[col for col in grids.columns if col.startswith('stop')]]

grid_steps = grids.delta.values



targets = start_grids.values.reshape(-1,25,25,1)

features = stop_grids.values.reshape(-1,25,25,1)

grid_steps.shape, targets.shape, features.shape
inputs = Input((25,25,1))



step_in = Input(1,)

step_out = Embedding(input_dim = 6, output_dim = 625, input_length = 1)(step_in)

step_out = Reshape((25,25,1))(step_out)



x1 = Conv2D(128,3,activation = None,padding = 'same', name = 'conv_size_3')(inputs)

bn1 = BatchNormalization()(x1)

bn1 = Activation('elu')(bn1)

x2 = Conv2D(32,4,activation = None,padding = 'same',name = 'conv_size_4')(inputs)

bn2 = BatchNormalization()(x2)

bn2 = Activation('elu')(bn2)

x3 = Conv2D(10,5,activation = None,padding = 'same',name = 'conv_size_5')(inputs)

bn3 = BatchNormalization()(x3)

bn3 = Activation('elu')(bn3)

x4 = Conv2D(10,7,activation = None,padding = 'same',name = 'conv_size_7')(inputs)

bn4 = BatchNormalization()(x4)

bn4 = Activation('elu')(bn4)

x5 = Conv2D(10,9,activation = None,padding = 'same',name = 'conv_size_9')(inputs)

bn5 = BatchNormalization()(x5)

bn5 = Activation('elu')(bn5)





x = Concatenate(axis = -1)([bn1, bn2, bn3, bn4, bn5])



x = Conv2D(32,3, activation = 'elu', padding = 'same', name = 'conv1_out_1')(x)

x = Multiply()([step_out, x])

x = Conv2D(32,3, activation = 'elu', padding = 'same', name = 'conv1_out_2')(x)

x = Multiply()([step_out, x])

x = Conv2D(32,3, activation = 'elu', padding = 'same', name = 'conv1_out_3')(x)

x = Multiply()([step_out, x])

x = Conv2D(32,3, activation = 'elu', padding = 'same', name = 'conv1_out_4')(x)

x = Multiply()([step_out, x])

x = Conv2D(32,3, activation = 'elu', padding = 'same', name = 'conv1_out_5')(x)

x = Multiply()([step_out, x])



x = Conv2D(1,3, activation = 'sigmoid', padding = 'same', name = 'conv1_out_final')(x)

model = Model([inputs,step_in], x)

model.summary()
model.compile(loss = 'binary_crossentropy',optimizer = Adam(lr = 0.001),metrics = ['accuracy'])

model.fit(x = [features, grid_steps],y = targets, epochs = 140, validation_split = 0.05, batch_size = 128)
test_grids = pd.read_csv('/kaggle/input/conways-reverse-game-of-life-2020/test.csv')

stop_grids_test = test_grids[[col for col in grids.columns if col.startswith('stop')]]

grid_steps_test = test_grids.delta.values

features_test = stop_grids_test.values.reshape(-1,25,25,1)

test_predictions = model.predict(x = [features_test, grid_steps_test])
submissions = pd.DataFrame((test_predictions.reshape(-1,625) > 0.5).astype(int), columns = [f'start_{i}'for i in np.arange(625)])

submissions['id'] = test_grids.id.values
submissions.to_csv("submission.csv", index=False)
submissions.head()