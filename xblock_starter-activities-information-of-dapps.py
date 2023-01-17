# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# 读取两个市场的Dapps数据到两个DataFrame中，注意使用正确的编码../input/Activity Information of DApps/

radar_df = pd.read_csv('../input/activities-information-of-dapps/Activity Information of DApps/radar.csv', encoding = 'gb18030')

state_df = pd.read_csv('../input/activities-information-of-dapps/Activity Information of DApps/state_of_the_dapp.csv', encoding = 'gb18030')
# Radar数据集属性和数据类型

radar_df.info()
# State of the DApps数据集属性和数据类型

state_df.info()
# Radar数据集描述

radar_df.describe()
# State of the DApps数据集描述

state_df.describe()
# Radar数据集前五行展示

radar_df.head()
# Radar数据集前五行展示

state_df.head()
# 本数据集可以用于可视化

# 以State of the Dapps数据集为例展示数据集的使用示例

df = state_df

df.index = df['rank']  # 以rank作为数据索引
# 显示数据在platform上的分布

sns.countplot(df['platform'])
# 过去24小时内活跃用户数量的和密度估计曲线

sns.distplot(df[df['users_24h']!='-']['users_24h'].astype('int'), hist=False)
# 过去30天内活跃合约数量的和密度估计曲线

sns.distplot(df[df['dev_activity_30d']!='-']['dev_activity_30d'].astype('int'), hist=False)