# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib.pyplot import savefig
from matplotlib import style
style.use("ggplot")

animals = pd.read_csv("../input/train.csv")

AnimalType = animals['AnimalType'].value_counts() 

AnimalType.plot(kind='bar',color='#348ABD',rot=0)
savefig("AnimalType.png")






# Any results you write to the current directory are saved as output.