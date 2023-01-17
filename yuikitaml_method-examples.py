import pandas as pd
import numpy as np
import re
import sklearn
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                              GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC
from sklearn.cross_validation import KFold
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
PassengerId = test['PassengerId']
train.head(3)
# Count NaN
train.Cabin.isnull().sum()
# Numerise Sex
result = train.Sex.map({'male': 0, 'female': 1})
result.head(3)
# Fill na
result = train.Cabin.fillna('S')
result.head(3)
