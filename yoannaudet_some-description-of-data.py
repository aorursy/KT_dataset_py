# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns
raw_data = pd.read_csv("/kaggle/input/ciri422/training.csv", header=None)

test_data = pd.read_csv("/kaggle/input/ciri422/test.csv", header=None)

output = pd.read_csv("/kaggle/input/ciri422/sample.csv")



Input = raw_data.index

Columns = raw_data.columns

output.columns = ["Id", "Redshift"]

raw_data.columns = ["ID", "U", "G", "R", "I", "Z", "Y", "Red-shift"]

test_data.columns = ["ID", "U", "G", "R", "I", "Z", "Y"]



red_shift = raw_data["Red-shift"]

raw_data = raw_data.drop(columns=["Red-shift", "ID"])

test_data = test_data.drop(columns=["ID"])



print("raw data (for training) : \n", raw_data.head())

print("Test data (to use for predictions) : \n", test_data.head())

print("Example of submition file : \n", output.head())
sns.pairplot(raw_data)

plt.show()


for i in raw_data.columns:

    plt.figure()

    sns.scatterplot(raw_data[i], red_shift)

    plt.show()

for i in raw_data.columns:

    sns.violinplot(raw_data[i])

    

plt.show()