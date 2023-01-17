!pip install pycaret
import numpy as np
import pandas as pd
import os, sys
from IPython.display import display

from pycaret.utils import version
from pycaret.datasets import get_data

print('Boston Data')
boston = get_data('boston')
from pycaret.regression import *
reg = setup(data = boston, target = 'medv')

compare_models()
cat=create_model('catboost')
print(cat)
tuned_cat=tune_model(cat)
print(tuned_cat)
ada = create_model('ada')
print(ada)
dt = create_model('dt')
print(dt)
lightgbm = create_model('lightgbm')