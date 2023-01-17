
import pandas as pd
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn, sklearn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
sns.set_style('whitegrid')
##%matplotlib inline



#### import the data
train   = pd.read_csv('../input/train.csv')
test    = pd.read_csv('../input/test.csv')
sample_sub = pd.read_csv('../input/sample_submission.csv')
train.head()
