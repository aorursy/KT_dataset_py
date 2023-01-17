!pip install pycaret
!pip install --upgrade pycaret #if you have installed beta version in past, run the below code to upgrade

import numpy as np
import pandas as pd
import os, sys
from IPython.display import display

from pycaret.utils import version
from pycaret.datasets import get_data

print('Diabetes Data')
diabetes = get_data('diabetes')

print('Boston Data')
boston = get_data('boston')
from pycaret.classification import *
clf = setup(data = diabetes, target = 'Class variable')

compare_models()
from pycaret.regression import *
reg = setup(data = boston, target = 'medv')

compare_models()