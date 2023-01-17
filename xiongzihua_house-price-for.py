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
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")

y_train = np.log(train_df['SalePrice']+1)

train_df.drop(['SalePrice'], axis=1, inplace=True)

all_data = pd.concat((train_df.loc[:,'Id':'SaleCondition'],

                      test_df.loc[:,'Id':'SaleCondition']))

all_data.head(3)
all_data[(all_data.PoolArea>0)&(all_data.PoolQC.isnull())][['Id','PoolArea','PoolQC']]
all_data.groupby('PoolQC',as_index=False)[['PoolArea']].mean()
len(all_data[all_data.GarageYrBlt==all_data.YearBuilt])
all_data[(all_data.GarageCond.isnull())&(all_data.GarageArea>0)][['Id','GarageArea', 'GarageCars', 'GarageQual', 'GarageFinish', 'GarageCond', 'GarageType']]
all_data[(all_data.Id==2127)]['']
train_df[train_df.Electrical.isnull()]