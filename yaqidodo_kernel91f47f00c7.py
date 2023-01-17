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
import pandas as pd
train_set = pd.read_csv("../input/data_train.csv")

train_set.head(30)
print(train_set.info())
vmax_0 = train_set.loc[(train_set['vmax'] == 0)]

vmax_0.head()

vmax_0
same_point = train_set.loc[(train_set['time_entry'] == train_set['time_exit'])]
same_point.rename(columns={'Unnamed: 0':'order'}, inplace=True)
same_point.index = range(len(same_point.index))

same_point
new_data = pd.DataFrame()
for index in range(433076):

    if same_point.loc[index].loc["hash"]== same_point.loc[index+1].loc["hash"]:

        print(index)

        if same_point.loc[index].loc["order"] == (same_point.loc[index+1].loc["order"]-1):

            new_row =pd.DataFrame([same_point.loc[index].loc["order"],same_point.loc[index].loc["hash"],same_point.loc[index].loc["trajectory_id"],  

                       same_point.loc[index].loc["time_entry"],same_point.loc[index+1].loc["time_exit"],

                       same_point.loc[index].loc["vmax"],same_point.loc[index].loc["vmin"],same_point.loc[index].loc["vmean"],

                       same_point.loc[index].loc["x_entry"],same_point.loc[index].loc["y_entry"],

                       same_point.loc[index+1].loc["x_entry"],same_point.loc[index+1].loc["y_entry"]])

            new_data= pd.concat([new_data,new_row.T],ignore_index=True)

        

            
# 
# index=2

# if same_point.loc[index].loc["hash"]== same_point.loc[index+1].loc["hash"]:

#     if same_point.loc[index].loc["order"] == same_point.loc[index+1].loc["order"]-1:

#         new_row =pd.DataFrame([same_point.loc[index].loc["order"],same_point.loc[index].loc["hash"],same_point.loc[index].loc["trajectory_id"],  

#                    same_point.loc[index].loc["time_entry"],same_point.loc[index+1].loc["time_exit"],

#                    same_point.loc[index].loc["vmax"],same_point.loc[index].loc["vmin"],same_point.loc[index].loc["vmean"],

#                    same_point.loc[index].loc["x_entry"],same_point.loc[index].loc["y_entry"],

#                    same_point.loc[index+1].loc["x_entry"],same_point.loc[index+1].loc["y_entry"]])

#         new_data= pd.concat([new_data.T,new_row.T],ignore_index=True)



new_data

new_data.to_csv("../working/submit.csv", index=False)