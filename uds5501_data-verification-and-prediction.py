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
df_train = pd.read_csv('../input/train_V2.csv', nrows = 100000)
df_test = pd.read_csv('../input/test_V2.csv', nrows = 100000)
df_train.head()
# importing important libraries
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# utility function 1
def mean_dev_stuff(df, column_name, limit = 20000):
    if len(df) > limit:
        df = df[:limit]
    
    data_bar = df[column_name].mean()
    df[column_name+"_squaredmean"] = (df[column_name] - data_bar)**2
    std_dev = (sum(df[column_name+"_squaredmean"])/len(df))**0.5
    
    UCL = np.array([data_bar + 3 * std_dev]*len(df))
    LCL = np.array([data_bar - 3 * std_dev]*len(df))
    mean_line = np.array([data_bar]*len(df))
    
    new_df = pd.DataFrame({column_name : df[column_name], "UCL" : UCL, "LCL" : LCL, "Mean" : mean_line})
    return new_df
# killing definition function
killdf_train = mean_dev_stuff(df_train, 'kills', 20000)
killdf_test = mean_dev_stuff(df_test, 'kills', 20000)
# kills by a player

plt.figure(figsize = (25, 10))
plt.subplot(2,2,1)
plt.plot(killdf_train.index ,killdf_train['kills'], color = "pink", linewidth = 1, label = "Kills by players")
plt.plot(killdf_train.index, killdf_train['UCL'], color = "red", linewidth = 2, linestyle = "--", label = "UCL" )
plt.plot( killdf_train.index, killdf_train['Mean']  , color = "blue", linewidth = 2, linestyle = "--", label = "Mean")

plt.title("The Kill Records of first 20,000 players in training", fontsize = 16)
plt.xlabel("Player Index")
plt.legend(prop={'size':16})
plt.tick_params(labelsize=16)

plt.subplot(2,2,2)
plt.plot(killdf_test.index ,killdf_test['kills'], color = "orange", linewidth = 1, label = "Kills by players")
plt.plot(killdf_test.index, killdf_test['UCL'], color = "red", linewidth = 2, linestyle = "--", label = "UCL")
plt.plot(killdf_test.index, killdf_test['Mean']  , color = "blue", linewidth = 2, linestyle = "--", label = "Mean")

plt.title("The Kill Records of first 20,000 players in testing", fontsize = 16)
plt.xlabel("Player Index")
plt.legend(prop={'size':16})
plt.tick_params(labelsize=16)

mean_df_train = killdf_train['Mean'].iloc[0]
mean_df_test = killdf_test['Mean'].iloc[0]
std_dev_train = (killdf_train['UCL'].iloc[0] - mean_df_train)/3
std_dev_test = (killdf_test["UCL"].iloc[0] - mean_df_test)/3

print("Mean in train : {:.5f}, Mean in test : {:.5f}".format(mean_df_train, mean_df_test))
print("Standard Deviation in train : {:.5f}, Standard Deviation in test : {:.5f}".format(std_dev_train, std_dev_test))
damagedf_train = mean_dev_stuff(df_train, 'damageDealt')
damagedf_test = mean_dev_stuff(df_test, 'damageDealt')
# damage dealt by a player

plt.figure(figsize = (25, 10))
plt.subplot(2,2,1)
plt.plot(damagedf_train.index ,damagedf_train['damageDealt'], color = "pink", linewidth = 1, label = "Damage dealt by players")
plt.plot(damagedf_train.index, damagedf_train['UCL'], color = "red", linewidth = 2, linestyle = "--", label = "UCL" )
plt.plot( damagedf_train.index, damagedf_train['Mean']  , color = "blue", linewidth = 2, linestyle = "--", label = "Mean")

plt.title("The Damage Dealings of first 20,000 players in training", fontsize = 16)
plt.xlabel("Player Index")
plt.legend(prop={'size':16})
plt.tick_params(labelsize=16)

plt.subplot(2,2,2)
plt.plot(damagedf_test.index ,damagedf_test['damageDealt'], color = "orange", linewidth = 1, label = "Damage dealt by players")
plt.plot(damagedf_test.index, damagedf_test['UCL'], color = "red", linewidth = 2, linestyle = "--", label = "UCL")
plt.plot(damagedf_test.index, damagedf_test['Mean']  , color = "blue", linewidth = 2, linestyle = "--", label = "Mean")

plt.title("The Damage Dealings of first 20,000 players in testing", fontsize = 16)
plt.xlabel("Player Index")
plt.legend(prop={'size':16})
plt.tick_params(labelsize=16)

mean_df_train = damagedf_train['Mean'].iloc[0]
mean_df_test = damagedf_test['Mean'].iloc[0]
std_dev_train = (damagedf_train['UCL'].iloc[0] - mean_df_train)/3
std_dev_test = (damagedf_test["UCL"].iloc[0] - mean_df_test)/3

print("Mean in train : {:.5f}, Mean in test : {:.5f}".format(mean_df_train, mean_df_test))
print("Standard Deviation in train : {:.5f}, Standard Deviation in test : {:.5f}".format(std_dev_train, std_dev_test))
# utility for encoding
df_train[(df_train['damageDealt'] > 0) & (df_train['kills'] <= df_train['DBNOs'])][['damageDealt', 'kills', 'DBNOs']].head(20)
print("MIN : {}, MAX : {}, MEAN : {}".format(min(df_train['matchDuration']), max(df_train['matchDuration']), df_train['matchDuration'].mean()))
study_df = pd.read_csv('../input/train_V2.csv', nrows = 1000000)
study_df.info()
Top10 = study_df[study_df['matchDuration'] >= 0.9]
Top10.head()
Top10.columns[3:]
x = ['assists', 'boosts', 'damageDealt', 'DBNOs', 'headshotKills', 'heals',
       'killPlace', 'killPoints', 'kills', 'killStreaks', 'longestKill',
       'matchDuration', 'maxPlace', 'numGroups', 'rankPoints',
       'revives', 'rideDistance', 'roadKills', 'swimDistance', 'teamKills',
       'vehicleDestroys', 'walkDistance', 'weaponsAcquired', 'winPoints',
       'winPlacePerc']
Top10.groupby(['matchType'], axis = 0)[x].mean()

