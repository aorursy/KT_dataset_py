import numpy as np

import pandas as pd

import keras

import keras.backend as K

from keras.optimizers import Adam

from keras.models import Sequential

from keras.utils import Sequence

from keras.layers import *

import matplotlib.pyplot as plt
path = "../input/sudoku/"

data = pd.read_csv(path+"sudoku.csv")

try:

    data = pd.DataFrame({"quizzes":data["puzzle"],"solutions":data["solution"]})

except:

    pass

data.head()
data.info()
print("Quiz:\n",np.array(list(map(int,list(data['quizzes'][0])))).reshape(9,9))

print("Solution:\n",np.array(list(map(int,list(data['solutions'][0])))).reshape(9,9))
#Utility Functions

class DataGenerator(Sequence):

    def __init__(self, df,batch_size = 16,subset = "train",shuffle = False, info={}):

        super().__init__()

        self.df = df

        self.batch_size = batch_size

        self.shuffle = shuffle

        self.subset = subset

        self.info = info

        

        self.data_path = path

        self.on_epoch_end()

        

    def __len__(self):

        return int(np.floor(len(self.df)/self.batch_size))

    def on_epoch_end(self):

        self.indexes = np.arange(len(self.df))

        if self.shuffle==True:

            np.random.shuffle(self.indexes)

            

    def __getitem__(self,index):

        X = np.empty((self.batch_size, 9,9,1))

        y = np.empty((self.batch_size,81,1))

        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        for i,f in enumerate(self.df['quizzes'].iloc[indexes]):

            self.info[index*self.batch_size+i]=f

            X[i,] = (np.array(list(map(int,list(f)))).reshape((9,9,1))/9)-0.5

        if self.subset == 'train': 

            for i,f in enumerate(self.df['solutions'].iloc[indexes]):

                self.info[index*self.batch_size+i]=f

                y[i,] = np.array(list(map(int,list(f)))).reshape((81,1)) - 1

        if self.subset == 'train': return X, y

        else: return X
model = Sequential()



model.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='same', input_shape=(9,9,1)))

model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'))

model.add(BatchNormalization())

model.add(Conv2D(128, kernel_size=(1,1), activation='relu', padding='same'))



model.add(Flatten())

model.add(Dense(81*9))

model.add(Reshape((-1, 9)))

model.add(Activation('softmax'))



adam = keras.optimizers.adam(lr=.001)

model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
model.summary()
train_idx = int(len(data)*0.95)

data = data.sample(frac=1).reset_index(drop=True)

training_generator = DataGenerator(data.iloc[:train_idx], subset = "train", batch_size=640)

validation_generator = DataGenerator(data.iloc[train_idx:], subset = "train",  batch_size=640)
training_generator.__getitem__(4)[0].shape
from keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau

filepath1="weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"

filepath2 = "best_weights.hdf5"

checkpoint1 = ModelCheckpoint(filepath1, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

checkpoint2 = ModelCheckpoint(filepath2, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')



reduce_lr = ReduceLROnPlateau(

    monitor='val_loss',

    patience=3,

    verbose=1,

    min_lr=1e-6

)

callbacks_list = [checkpoint1,checkpoint2,reduce_lr]
history = model.fit_generator(training_generator, validation_data = validation_generator, epochs = 1, verbose=1,callbacks=callbacks_list )
model.load_weights('best_weights.hdf5')
def norm(a):

    return (a/9)-.5



def denorm(a):

    return (a+.5)*9



def inference_sudoku(sample):

    

    '''

        This function solve the sudoku by filling blank positions one by one.

    '''

    

    feat = sample

    

    while(1):

    

        out = model.predict(feat.reshape((1,9,9,1)))  

        out = out.squeeze()



        pred = np.argmax(out, axis=1).reshape((9,9))+1 

        prob = np.around(np.max(out, axis=1).reshape((9,9)), 2) 

        

        feat = denorm(feat).reshape((9,9))

        mask = (feat==0)

     

        if(mask.sum()==0):

            break

            

        prob_new = prob*mask

    

        ind = np.argmax(prob_new)

        x, y = (ind//9), (ind%9)



        val = pred[x][y]

        feat[x][y] = val

        feat = norm(feat)

    

    return pred



def test_accuracy(feats, labels):

    

    correct = 0

    

    for i,feat in enumerate(feats):

        

        pred = inference_sudoku(feat)

        

        true = labels[i].reshape((9,9))+1

        

        if(abs(true - pred).sum()==0):

            correct += 1

        

    print(correct/feats.shape[0])



def solve_sudoku(game):

    

    game = game.replace('\n', '')

    game = game.replace(' ', '')

    game = np.array([int(j) for j in game]).reshape((9,9,1))

    game = norm(game)

    game = inference_sudoku(game)

    return game
new_game = '''

          0 0 0 0 0 0 0 0 0

          0 0 0 0 0 0 0 0 0

          0 0 0 0 0 0 0 0 0

          0 0 0 0 0 0 0 0 0

          0 0 0 0 0 0 0 0 0

          0 0 0 0 0 0 0 0 0

          0 0 0 0 0 0 0 0 0

          0 0 0 0 0 0 0 0 0

          0 0 0 0 0 0 0 0 0

      '''



game = '''

          0 0 0 7 0 0 0 9 6

          0 0 3 0 6 9 1 7 8

          0 0 7 2 0 0 5 0 0

          0 7 5 0 0 0 0 0 0

          9 0 1 0 0 0 3 0 0

          0 0 0 0 0 0 0 0 0

          0 0 9 0 0 0 0 0 1

          3 1 8 0 2 0 4 0 7

          2 4 0 0 0 5 0 0 0

      '''



game = solve_sudoku(game)



print('solved puzzle:\n')

print(game)
np.sum(game, axis=1)
def solve(bo):

    find = find_empty(bo)

    if not find:

        return True

    else:

        row, col = find



    for i in range(1,10):

        if valid(bo, i, (row, col)):

            bo[row][col] = i



            if solve(bo):

                return True



            bo[row][col] = 0



    return False





def valid(bo, num, pos):

    # Check row

    for i in range(len(bo[0])):

        if bo[pos[0]][i] == num and pos[1] != i:

            return False



    # Check column

    for i in range(len(bo)):

        if bo[i][pos[1]] == num and pos[0] != i:

            return False



    # Check box

    box_x = pos[1] // 3

    box_y = pos[0] // 3



    for i in range(box_y*3, box_y*3 + 3):

        for j in range(box_x * 3, box_x*3 + 3):

            if bo[i][j] == num and (i,j) != pos:

                return False



    return True





def print_board(bo):

    for i in range(len(bo)):

        if i % 3 == 0 and i != 0:

            print("- - - - - - - - - - - - - ")



        for j in range(len(bo[0])):

            if j % 3 == 0 and j != 0:

                print(" | ", end="")



            if j == 8:

                print(bo[i][j])

            else:

                print(str(bo[i][j]) + " ", end="")





def find_empty(bo):

    for i in range(len(bo)):

        for j in range(len(bo[0])):

            if bo[i][j] == 0:

                return (i, j)  # row, col



    return None
%%time

game = '''

          0 0 0 7 0 0 0 9 6

          0 0 3 0 6 9 1 7 8

          0 0 7 2 0 0 5 0 0

          0 7 5 0 0 0 0 0 0

          9 0 1 0 0 0 3 0 0

          0 0 0 0 0 0 0 0 0

          0 0 9 0 0 0 0 0 1

          3 1 8 0 2 0 4 0 7

          2 4 0 0 0 5 0 0 0

      '''

game = game.strip().split("\n")

board = []

for i in game:

    t = i.replace(' ','').strip()

    t = list(t)

    t = list(map(int,t))

    board.append(t)

    

if solve(board):

    print_board(board)

else:

    print("Can't be solved.")
np.sum(board, axis=1)
val_set = data.iloc[:1000]





from tqdm import tqdm

quiz_list = list(val_set['quizzes'])

sol_list = list(val_set['solutions'])

val_quiz = []

val_sol = []

for i,j in tqdm(zip(quiz_list,sol_list)):

    q = np.array(list(map(int,list(i)))).reshape(9,9)

    s = np.array(list(map(int,list(j)))).reshape(9,9)

    val_quiz.append(q)

    val_sol.append(s)
%%time

count = 0

for i,j in tqdm(zip(val_quiz,val_sol)):

    if solve(i):

        if (i==j).all():

            count+=1

    else:

        pass

    

print("{}/1000 solved!! That's {}% accuracy.\n".format(count,(count/1000.0)*100))
import numpy as np

import pandas as pd



import collections



rows = 'ABCDEFGHI'

cols = '123456789'



def cross(A, B):

    "Cross product of elements in A and elements in B."

    return [s + t for s in A for t in B]





boxes = cross(rows, cols)



row_units = [cross(r, cols) for r in rows]

column_units = [cross(rows, c) for c in cols]

square_units = [cross(rs, cs) for rs in ('ABC', 'DEF', 'GHI') for cs in ('123', '456', '789')]

unitlist = row_units + column_units + square_units 

units = dict((s, [u for u in unitlist if s in u]) for s in boxes)

peers = dict((s, set(sum(units[s], [])) - set([s])) for s in boxes)





def assign_value(values, box, value):

    """

    Please use this function to update your values dictionary!

    Assigns a value to a given box. If it updates the board record it.

    """

    values[box] = value

    return values





def naked_twins(values):

    """Eliminate values using the naked twins strategy.

    Args:

        values(dict): a dictionary of the form {'box_name': '123456789', ...}

    Returns:

        the values dictionary with the naked twins eliminated from peers.

    """



    # Find all instances of naked twins

    for unit in unitlist:

        # Occurrences dict

        unit_values_counter = collections.Counter([values[box] for box in unit])

        for twins, count in unit_values_counter.items():

            # twins will occur twice in a unit, triples will occur three times, and quads four times

            if 1 < count == len(twins):

                for box in unit:

                    # for all boxes except twins boxes in a unit,

                    # remove all potential values that exist in twins, triples, quads..

                    if values[box] != twins and set(values[box]).intersection(set(twins)):

                        for digit in twins:

                            values = assign_value(values, box, values[box].replace(digit, ''))

    return values





def grid_values(grid):

    """

    Convert grid into a dict of {square: char} with '123456789' for empties.

    Args:

        grid(string) - A grid in string form.

    Returns:

        A grid in dictionary form

            Keys: The boxes, e.g., 'A1'

            Values: The value in each box, e.g., '8'. If the box has no value, then the value will be '123456789'.

    """

    chars = []

    digits = '123456789'

    for c in grid:

        if c in digits:

            chars.append(c)

        if c == '0':

            chars.append(digits)

    assert len(chars) == 81

    return dict(zip(boxes, chars))





def display(values):

    """

    Display the values as a 2-D grid.

    Args:

        values(dict): The sudoku in dictionary form

    """

    width = 1 + max(len(values[s]) for s in boxes)

    line = '+'.join(['-' * (width * 3)] * 3)

    for r in rows:

        print(''.join(values[r + c].center(width) + ('|' if c in '36' else '')

                      for c in cols))

        if r in 'CF': print(line)

    print





def eliminate(values):

    """

        Go through all the boxes, and whenever there is a box with a value, eliminate this value from the values of all its peers.

        Input: A sudoku in dictionary form.

        Output: The resulting sudoku in dictionary form.

        """

    solved_values = [box for box in values.keys() if len(values[box]) == 1]

    for box in solved_values:

        digit = values[box]

        for peer in peers[box]:

            values[peer] = values[peer].replace(digit, '')

    return values





def only_choice(values):

    """

        Go through all the units, and whenever there is a unit with a value that only fits in one box, assign the value to this box.

        Input: A sudoku in dictionary form.

        Output: The resulting sudoku in dictionary form.

        """

    for unit in unitlist:

        for digit in '123456789':

            dplaces = [box for box in unit if digit in values[box]]

            if len(dplaces) == 1:

                values[dplaces[0]] = digit

    return values





def reduce_puzzle(values):

    """

    Iterate eliminate() and only_choice(). If at some point, there is a box with no available values, return False.

    If the sudoku is solved, return the sudoku.

    If after an iteration of both functions, the sudoku remains the same, return the sudoku.

    Input: A sudoku in dictionary form.

    Output: The resulting sudoku in dictionary form.

    """

    stalled = False

    while not stalled:

        solved_values_before = len([box for box in values.keys() if len(values[box]) == 1])

        values = eliminate(values)

        values = only_choice(values)

        values = naked_twins(values)

        solved_values_after = len([box for box in values.keys() if len(values[box]) == 1])

        stalled = solved_values_before == solved_values_after

        if len([box for box in values.keys() if len(values[box]) == 0]):

            #display(values)

            return False

    return values





def search(values):

    "Using depth-first search and propagation, create a search tree and solve the sudoku."

    # First, reduce the puzzle using the previous function

    values = reduce_puzzle(values)

    if values is False:

        return False  ## Failed earlier

    if all(len(values[s]) == 1 for s in boxes):

        return values  ## Solved!

    # Choose one of the unfilled squares with the fewest possibilities

    min_possibility_box = min([box for box in boxes if len(values[box]) > 1])

    # Now use recursion to solve each one of the resulting sudokus, and if one returns a value (not False), return that answer!

    for digit in values[min_possibility_box]:

        new_sudoku = values.copy()

        new_sudoku[min_possibility_box] = digit

        attempt = search(new_sudoku)

        if attempt:

            return attempt





def solve2(grid):



    values = grid_values(grid)

    values = search(values)

    return values
%%time

count = 0

for row in tqdm(data.head(1000).iterrows()):

    if (solve2(row[1]["quizzes"]) == grid_values(row[1]["solutions"])):

        count+=1

        

print("{}/1,000 solved!! That's {}% accuracy.\n".format(count,(count/1000.0)*100))
%%time

from multiprocessing import Pool

num_partitions = 100 #number of partitions to split dataframe

num_cores = 6 #number of cores on your machine



def parallelize_dataframe(df, func):

    df_split = np.array_split(df, num_partitions)

    pool = Pool(num_cores)

    pool.map(func, df_split)

    pool.close()

    pool.join()



def solve_and_verify(data):

    for row in data.iterrows():

        assert solve2(row[1]["quizzes"]) == grid_values(row[1]["solutions"])

    

parallelize_dataframe(data.head(1000), solve_and_verify)