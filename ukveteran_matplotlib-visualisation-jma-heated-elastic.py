import pandas as pd 

import numpy as np 

import matplotlib.pyplot as plt 

%matplotlib inline

import random

import seaborn as sns

from fbprophet import Prophet

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
df = pd.read_csv('../input/heated-elastic-bands/pair65.csv')

df.head()
cm = sns.light_palette("blue", as_cmap=True)



s = df.style.background_gradient(cmap=cm)

s