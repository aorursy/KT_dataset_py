from typing import List

from os.path import join as pjoin



import numpy as np

import pandas as pd



from matplotlib import pyplot as plt



%matplotlib inline
path = '../input/stanford-covid-vaccine/'

train = pd.read_json(f'{path}/train.json',lines=True).drop(columns='index')

test = pd.read_json(f'{path}/test.json', lines=True).drop(columns='index')

sub = pd.read_csv(f'{path}/sample_submission.csv')
STRUCTURE_CODE = {

    '(': 0, 

    '.': 1, 

    ')': 2

}



PREDICTED_LOOP_TYPE_CODE = {

    'H': 0, 

    'E': 1, 

    'B': 2, 

    'M': 3, 

    'X': 4, 

    'S': 5, 

    'I': 6

}



SEQUANCE_CODE = {

    'U': 0, 

    'C': 1, 

    'A': 2, 

    'G': 3

}



X_COLS = ['sequence', 'structure', 'predicted_loop_type']

X_MAPPINGS = [SEQUANCE_CODE, STRUCTURE_CODE, PREDICTED_LOOP_TYPE_CODE]

Y_COLS = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']
def get_x_y_atten(

    df_row: pd.Series,

    root: str = pjoin(path,'bpps')

):

    x_array = []

    for col, mapping in zip(X_COLS, X_MAPPINGS):

        x_array.append(np.array([mapping[el] for el in df_row[col]]))

    x_array = np.stack(x_array, axis=-1)

    

    y_array = [np.array(df_row[col]) for col in Y_COLS]

    y_array = np.stack(y_array, axis=-1)

    

    bpps = np.load(pjoin(root, df_row.id + '.npy'))

    

    return x_array, y_array, bpps
class Compose(object):

    

    def __init__(

        self,

        transforms: List

    ):

        self.transforms = transforms

        

    def __call__(self, x_arr, y_arr, atten_arr):

        for trans in self.transforms:

            x_arr, y_arr, atten_arr = trans(x_arr, y_arr, atten_arr)

        return x_arr, y_arr, atten_arr
class TemporalFlip(object):

    

    def __init__(

        self,

        p: float = 0.5

    ):

        self.p = p

        

    def __call__(self, x_arr, y_arr, atten_arr):

        

        if np.random.binomial(n=1, p=self.p):

            x_arr = np.flip(x_arr, axis=0).copy()

            y_arr = np.flip(y_arr, axis=0).copy()

            atten_arr = np.flip(np.flip(atten_arr, axis=0), axis=1).copy()

            

        return x_arr, y_arr, atten_arr
flip_aug = TemporalFlip(p=1.0)
x, y, atten = get_x_y_atten(train.iloc[0])

x_a, y_a, atten_a = flip_aug(x, y, atten)
plt.title('Original Structure')

plt.plot(x[:,0])

plt.show()

plt.title('Auged Structure')

plt.plot(x_a[:,0])

plt.show()
plt.title('Original Sequence')

plt.plot(y[:,0])

plt.show()

plt.title('Auged Sequence')

plt.plot(y_a[:,0])

plt.show()
plt.title('Original bpps')

plt.imshow(atten)

plt.show()

plt.title('Auged bpps')

plt.imshow(atten_a)

plt.show()
class GaussianTargetNoise(object):

    

    def __init__(

        self,

        p: float = 0.5,

        gaus_std: float = 1.0,

    ):

        self.p = p

        self.gaus_std = gaus_std

        

    def __call__(self, x_arr, y_arr, atten_arr):

        

        if np.random.binomial(n=1, p=self.p):

            y_arr = y_arr + np.random.normal(scale=self.gaus_std, size=y_arr.shape)

            

        return x_arr, y_arr, atten_arr
gaus_aug = GaussianTargetNoise(p=1.0, gaus_std=0.3)
x, y, atten = get_x_y_atten(train.iloc[0])

x_a, y_a, atten_a = gaus_aug(x, y, atten)
for i in range(5):

    plt.title(Y_COLS[i])

    plt.plot(y[:,i], label='original')

    plt.plot(y_a[:,i], label='auged')

    plt.legend()

    plt.show()
combined_aug = Compose(transforms=[

    GaussianTargetNoise(p=1.0, gaus_std=0.3),

    TemporalFlip(p=1.0)

])
x, y, atten = get_x_y_atten(train.iloc[0])

x_a, y_a, atten_a = combined_aug(x, y, atten)
plt.title('Original Structure')

plt.plot(x[:,0])

plt.show()

plt.title('Auged Structure')

plt.plot(x_a[:,0])

plt.show()
plt.title('Original Sequence')

plt.plot(y[:,0])

plt.show()

plt.title('Auged Sequence')

plt.plot(y_a[:,0])

plt.show()
plt.title('Original bpps')

plt.imshow(atten)

plt.show()

plt.title('Auged bpps')

plt.imshow(atten_a)

plt.show()