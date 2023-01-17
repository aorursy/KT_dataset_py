# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/smart-contract-attribute-dataset/Smart Contract Attribute Dataset/OpenSourceContractInfo.csv')
df.head()
# Check attributes

df.columns
# Check contract address

df['address'][0]
# Check contract createValue and createdBlockNumber

print(df['createValue'][0])

print(df['createdBlockNumber'][0])
# Get timestamp

df['timestamp'][0]
# Get creator

df['creator'][0]
# Get creation transaction

df['createdTransactionHash'][0]