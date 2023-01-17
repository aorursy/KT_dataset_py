!pip install pycaret==2.0
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from pycaret.utils import version
from pycaret.datasets import get_data
dataset = get_data('iris')
dataset.shape
data = dataset.sample(frac=0.9, random_state=786).reset_index(drop=True)
data_unseen = dataset.drop(data.index).reset_index(drop=True)

print('Data for Modeling: ' + str(data.shape))
print('Unseen Data For Predictions: ' + str(data_unseen.shape))
from pycaret.classification import *
iris = setup(data = data, target = 'species', session_id=1)
compare_models()
naive = create_model('nb')
qda=create_model('qda')
xgb=create_model('xgboost')
cat=create_model('catboost')