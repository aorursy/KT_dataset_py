import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.optimize import minimize
# Let's generate some labels and 3 prediction sets.
L = (np.random.rand(1800, 206) > 0.5).astype(int)
p1 = np.random.rand(1800, 206)
p2 = np.random.rand(1800, 206)
p3 = np.random.rand(1800, 206)
# Helpers
def individual_scores(label, preds: list):
    for i, pred in enumerate(preds):
        print('LogLoss for p%d: %.5f'  % (i, tf.keras.losses.binary_crossentropy(L, pred).numpy().mean()))

def show_weights(result):
    print('\n')
    for i, w in enumerate(result.x):
        print('Weight_%d: %f' % (i, w))

def sanity_check():
    # All probabilities have to be between 0 and 1.
    if ((blend_func(res.x) > 0) & (blend_func(res.x) < 1)).all():
        print('\nAll probabilities are between 0 and 1. \n    Good to go!')
    else:
        print('\nProbabilities are not between 0 and 1! \nS    Something is wrong!')
        
# Optimization
def objective_func(x):
    newp = blend_func(x)
    return tf.keras.losses.binary_crossentropy(L, newp).numpy().mean()

def blend_func(x):
    return p1*x[0] + p2*x[1] + p3*x[2]


individual_scores(L, [p1, p2, p3])
init_guess = [0.1,0.5,0.4]  # Initial guesses for the weights
bounds = tuple((0,1) for x in init_guess)  # All weights will be between 0 and 1!
cons = ({'type': 'eq', 'fun': lambda x:  1-x[0]-x[1]-x[2]})  # Constraints - the sum of weights will be 0!

res = minimize(objective_func,
               init_guess,
               constraints=cons,
               bounds=[(0, 1)] * 3,
               method='SLSQP',
               options={'disp': True,
                        'maxiter': 100000}) # Minimize!
print('\nBlend LogLoss: %.5f' % (res.fun))
show_weights(res)
sanity_check()