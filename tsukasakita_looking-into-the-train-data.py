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
train_data = pd.read_csv('../input/train_V2.csv')
# this is a squad match
match_data = train_data[train_data["matchId"]==train_data["matchId"].iloc[102]]
print(match_data["matchType"].unique())
for group_id in match_data["groupId"].unique():
    members = len(match_data[match_data["groupId"]==group_id])
    if members > 4:
        print("groupId is %s" %group_id)
        print(match_data[match_data["groupId"]==group_id])
        print("%d members" %members)
        print()
