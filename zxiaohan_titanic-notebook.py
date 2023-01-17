# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os

%matplotlib inline



data_train = pd.read_csv('../input/train.csv')

data_test = pd.read_csv('../input/test.csv')

os.getcwd()

os.listdir('/kaggle/input')



data_train.sample(3)
sns.barplot(x="Embarked", y="Survived", hue="Sex", data=data_train);

sns.pointplot(x="Pclass", y="Survived", hue="Sex", data=data_train,

              palette={"male": "blue", "female": "pink"},

              markers=["*", "o"], linestyles=["-", "--"]);