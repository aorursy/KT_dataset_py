# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



## Load The Dataset

credit_cards_raw = pd.read_csv("../input/creditcard.csv")



## Exploration

credit_cards_raw.shape[0]