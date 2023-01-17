!pip install pandas-profiling[notebook,html]
!ls ../input/otto-group-product-classification-challenge
import pandas as pd



train = pd.read_csv("../input/otto-group-product-classification-challenge/train.csv")

test = pd.read_csv("../input/otto-group-product-classification-challenge/test.csv")
train.shape, test.shape
from pandas_profiling import ProfileReport
train_profile = ProfileReport(train, title='Pandas Profiling Report', html={'style':{'full_width':True}})
train_profile.to_notebook_iframe()