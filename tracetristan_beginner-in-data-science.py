import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
titanic=pd.read_csv("../input/train.csv")

titanic.info()

titanic.head()