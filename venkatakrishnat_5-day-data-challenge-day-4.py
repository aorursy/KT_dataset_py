# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import seaborn as sns
import pandas as pd
import os
import matplotlib.pyplot as plt
print(os.listdir("../input"))
cereal = pd.read_csv("../input/cereal.csv")
# print(cereal.head())
mfrFre = cereal["mfr"].value_counts()
list(mfrFre.index)
mfrFre.values
labels = list(mfrFre.index)
print(labels)

positionForBars = list(range(len(labels)))

plt.bar(positionForBars, mfrFre.values)
plt.xticks(positionForBars, labels)
plt.title("Cereal manufacturer")
import seaborn as sns
sns.countplot(cereal["mfr"]).set_title("Cereal manufacturer")