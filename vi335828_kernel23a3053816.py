#importing all the important lib.





import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import keras

import re

# Setting seed for reproducability

np.random.seed(1234)  

PYTHONHASHSEED = 0

from sklearn import preprocessing

from sklearn.metrics import confusion_matrix, recall_score, precision_score

from keras.models import Sequential

import glob

from keras.layers import Dense, Dropout, LSTM, Activation

%matplotlib inline
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
#combining all the samples into one dataframe 

path = r'/kaggle/input/one-year-industrial-component-degradation/' 

all_files = glob.glob(path + "/*.csv")

print (len(all_files))

data=[]

for i in all_files:

    

    #taking the unique mode value for example there are 8 mode so my dataframe will contain mode number to which that data point belong 

    file=re.split("_",i)

    mode=re.split(".csv",file[2])

    mode=re.search("[0-9]",mode[0])

    

    #taking all the unique sample values i.e 519 and datapoint will have respective sample no in dataframe 

    sample=re.split(".csv",file[1])

    

    #creating a dataframe 

    dataframe=pd.read_csv(i)

    dataframe["mode"]=mode.group(0)

    dataframe["sample"]=sample[0].lstrip("0")

    

    #appending it to the list 

    data.append(dataframe)

    

#concating a list to get final whole dataframe     

data_final=pd.concat(data,ignore_index=True)

print (data_final.shape)

# for taking sample no 0 

def sample_edit(data):

        return "0"

data_final["sample"].loc[0:2048]=data_final["sample"].head(2048).apply(sample_edit)

data_final.head()
# lest see distribution of datapoints with respect to mode 

data_final["mode"].value_counts()
#lets see distribution of sample with respect to mode 

data_final.groupby(["sample","mode"]).size().groupby("mode").count()

data_final["sample"].nunique()
data_final.head()
#renaming column name as we are having unneccessary lengthy names 

data_final.rename(columns={"pCut::Motor_Torque":"MT","pCut::CTRL_Position_controller::Lag_error": "Lag_error","pCut::CTRL_Position_controller::Actual_position" :"AP","pCut::CTRL_Position_controller::Actual_speed":"As","pSvolFilm::CTRL_Position_controller::Actual_position":"pAp","pSvolFilm::CTRL_Position_controller::Actual_speed":"pAs","pSvolFilm::CTRL_Position_controller::Lag_error":"ple","pSpintor::VAX_speed":"vax"},inplace=True)

#lets fill the null values

data_final.fillna(method="backfill",inplace=True)



#lets change the datatype of column name mode and sample

#data_final["sample"]=data_final["sample"].astype("int32")

data_final["mode"]=data_final["mode"].astype("int32")

all_sample=np.unique(data_final["sample"].values)

all_sample
#lets create cycle for every data point 



""" here i am creating a cycle for example lets take mode 1 and sample no 001 so we have 

    2048 total datapoints in this sample.

    so i will create a cycle starting from 1st datapoint to last datapoint.

    so for first my cycle value will be 1 because this is the first reading at timestamp 0.008 so i will increament my cycle 

    by 1 for every datapoint i.e for 2nd datapoint my cycle value will be 2 . 

    it will continue to increase by 1 till last point i.e 2048 so my final datapoint will be having cycle as 2048 or 2049

    if my data points starts from 0 or 1 """

data_final=data_final.assign(cycle=" ")

data_final.head()
#lets increment the cycle

for i in all_sample:

    i=i

    cycle=0

    for row in data_final.itertuples():

        sample=row[11]

        index=row[0]

        if i==sample:

            cycle +=1

            data_final.at[index,"cycle"]=cycle

           

            

            
data_final.head()
#lets create remaining useful life based on cycle



""" so lets take example of mode 1 and sample no 001.

    so we have total 2048 datapoints means 2048 cycle , so for first datapoint of mode 1 and sample 001 

    our mode 1 still having 2047 cycles are left or you can say still our mode 1 has 2048 readings .

    based on that i will create RUL(remaining useful life) like for first datapoint my RUL will be 2048-1 as there 

    are still 2048 reading or cycles needs to be complete, same thing i will repeat for remaining datapoints in 

    my data like for 2nd datapoint i will be having 2048-2 and it goes on."""



rul = pd.DataFrame(data_final.groupby('mode')['cycle'].max()).reset_index()

rul.columns = ['mode', 'max']

print (rul.head())

data_final = data_final.merge(rul, on=['mode'], how='left')

data_final['RUL'] =data_final['max'] - data_final['cycle']

data_final.drop('max', axis=1, inplace=True)

data_final.head()


data_final["RUL"]=data_final["RUL"].astype("int32")
#lets see how does our features behaves with timestamp

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns



plot_df = data_final.loc[(data_final['mode'] == 2) & 

                        (data_final['timestamp'] > 0.006) & 

                        (data_final['timestamp'] < 1.000),

                        ['timestamp','vax']]

sns.set_style("darkgrid")

plt.figure(figsize=(20, 8))

plt.plot(plot_df['timestamp'], plot_df['vax'])

plt.ylabel('voltage')
%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns



plot_df = data_final.loc[(data_final['mode'] == 4) & 

                        (data_final['timestamp'] > 0.006) & 

                        (data_final['timestamp'] < 1.000),

                        ['timestamp','pAs']]

sns.set_style("darkgrid")

plt.figure(figsize=(20, 8))

plt.plot(plot_df['timestamp'], plot_df['pAs'])

plt.ylabel('voltage')
#lets see how does our features behaves with timestamp

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns



plot_df = data_final.loc[(data_final['mode'] == 2) & 

                        (data_final['timestamp'] > 0.006) & 

                        (data_final['timestamp'] < 1.000),

                        ['timestamp','pAp']]

sns.set_style("darkgrid")

plt.figure(figsize=(20, 8))

plt.plot(plot_df['timestamp'], plot_df['pAp'])

plt.ylabel('voltage')
#Let us create a helper function to ease exploration of each feature invidually



def explore_col(s, e):

    

    """Plot 4 main graphs for a single feature.

    

        plot1: histogram 

        plot2: boxplot 

        plot3: line plot (time series over cycle)

        plot4: scatter plot vs. regression label ttf

        

    Args:

        s (str): The column name of the feature to be plotted.

        e (int): The number of random mode to be plotted for plot 3. Range from 1 -8 (becuase we have only 8 mode)



    Returns:

        plots

    

    """

    

    fig = plt.figure(figsize=(10, 8))

    sub1 = fig.add_subplot(221) 

    sub1.set_title(s +' histogram') 

    sub1.hist(data_final[s])



    sub2 = fig.add_subplot(222)

    sub2.set_title(s +' boxplot')

    sub2.boxplot(data_final[s])

    

    if e > 9 or e <= 0:

        select_mode = list(pd.unique(data_final.mode))

    else:

        select_mode = np.random.choice(range(0,9), e, replace=False)

        

    sub3 = fig.add_subplot(223)

    sub3.set_title('time series: ' + s +' / cycle')

    sub3.set_xlabel('cycle')

    for i in select_mode:

        df = data_final[['cycle', s]][data_final["mode"] == i]

        sub3.plot(df['cycle'],df[s])

        

    sub4 = fig.add_subplot(224)

    sub4.set_title("scatter: "+ s + " / RUL (regr label)")

    sub4.set_xlabel('RUL')

    sub4.scatter(data_final['RUL'],data_final[s])





    plt.tight_layout()

    plt.show()
explore_col("vax",4)
explore_col("vax",4)
# Create a function to explore the time series plot each sensor selecting random sample engines



def plot_time_series(s):

    

    """Plot time series of a single sensor for 10 random sample engines.

    

        Args:

        s (str): The column name of the sensor to be plotted.



    Returns:

        plots

        

    """

    

    fig, axes = plt.subplots(5, 1, sharex=True, figsize = (15, 15))

    fig.suptitle(s + ' time series / cycle', fontsize=15)

    

    #np.random.seed(12345)

    select_engines = np.random.choice(range(1,9), 5, replace=False).tolist()

    

    for e_id in select_engines:

        df = data_final[['cycle', s]][data_final["mode"]== e_id]

        i = select_engines.index(e_id)

        axes[i].plot(df['cycle'],df[s])

        axes[i].set_ylabel('mode ' + str(e_id))

        axes[i].set_xlabel('cycle')

        #axes[i].set_title('engine ' + str(e_id), loc='right')



    #plt.tight_layout()

    plt.subplots_adjust(wspace=0, hspace=0)

    plt.show()
plot_time_series('vax')
plot_time_series('AP')
#overall cycle



""" so lets take example of mode 1 and we have total 468992 datapoints for mode1 means 468992 cycle & we will increment by 1

    for every datapoint as we did for cycle for mode1 and sample 001"""



all_mode=np.unique(data_final["mode"].values)

data_final=data_final.assign(tot_cycle=" ")

for i in all_mode:

    i=int(i)

    cycle=0

    for row in data_final.itertuples():

        mode=row[10]

        index=row[0]

        if i==mode:

            cycle +=1

            data_final.at[index,"tot_cycle"]=cycle

           

            
#lets create remaining useful life based on tot_cycle



""" so lets take example of mode 1 my tot_rulL will be 468992-1 as there 

    are still 468992 reading or cycles needs to be complete, same thing i will repeat for remaining datapoints in 

    my data like for 2nd datapoint i will be having 468992-2 and it goes on and will do same for others mode also """



tot_rul = pd.DataFrame(data_final.groupby('mode')['tot_cycle'].max()).reset_index()

tot_rul.columns = ['mode', 'max']

print (tot_rul.head())

data_final = data_final.merge(tot_rul, on=['mode'], how='left')

data_final['Tot_RUL'] =data_final['max'] - data_final['tot_cycle']

data_final.drop('max', axis=1, inplace=True)

data_final.head()
#Let us create a helper function to ease exploration of each feature invidually



def explore_col(s, e):

    

    """Plot 4 main graphs for a single feature.

    

        plot1: histogram 

        plot2: boxplot 

        plot3: line plot (time series over cycle)

        plot4: scatter plot vs. regression label ttf

        

    Args:

        s (str): The column name of the feature to be plotted.

        e (int): The number of random mode to be plotted for plot 3. Range from 1 -8 (becuase we have only 8 mode)



    Returns:

        plots

    

    """

    

    fig = plt.figure(figsize=(10, 8))

    sub1 = fig.add_subplot(221) 

    sub1.set_title(s +' histogram') 

    sub1.hist(data_final[s])



    sub2 = fig.add_subplot(222)

    sub2.set_title(s +' boxplot')

    sub2.boxplot(data_final[s])

    

    if e > 9 or e <= 0:

        select_mode = list(pd.unique(data_final.mode))

    else:

        select_mode = np.random.choice(range(0,9), e, replace=False)

        

    sub3 = fig.add_subplot(223)

    sub3.set_title('time series: ' + s +' / cycle')

    sub3.set_xlabel('cycle')

    for i in select_mode:

        df = data_final[['tot_cycle', s]][data_final["mode"] == i]

        sub3.plot(df['tot_cycle'],df[s])

        

    sub4 = fig.add_subplot(224)

    sub4.set_title("scatter: "+ s + " / RUL (regr label)")

    sub4.set_xlabel('Tot_RUL')

    sub4.scatter(data_final['Tot_RUL'],data_final[s])





    plt.tight_layout()

    plt.show()


explore_col("pAs",2)
explore_col("vax",4)
# Create a function to explore the time series plot each sensor selecting random sample engines



def plot_time_series(s):

    

    """Plot time series of a single sensor for 10 random sample engines.

    

        Args:

        s (str): The column name of the sensor to be plotted.



    Returns:

        plots

        

    """

    

    fig, axes = plt.subplots(5, 1, sharex=True, figsize = (15, 15))

    fig.suptitle(s + ' time series / cycle', fontsize=15)

    

    #np.random.seed(12345)

    select_engines = np.random.choice(range(1,9), 5, replace=False).tolist()

    

    for e_id in select_engines:

        df = data_final[['tot_cycle', s]][data_final["mode"]== e_id]

        i = select_engines.index(e_id)

        axes[i].plot(df['tot_cycle'],df[s])

        axes[i].set_ylabel('mode ' + str(e_id))

        axes[i].set_xlabel('cycle')

        #axes[i].set_title('engine ' + str(e_id), loc='right')



    #plt.tight_layout()

    plt.subplots_adjust(wspace=0, hspace=0)

    plt.show()
plot_time_series('vax')
plot_time_series('AP')
#plot and compare the standard deviation of features:

features=['MT', 'Lag_error', 'AP', 'As', 'pAp', 'pAs', 'ple', 'vax']

data_final[features].std().plot(kind="bar",figsize=(8,6), title="Features Standard Deviation")
# get ordered list of top variance features:



featurs_top_var = data_final[features].std().sort_values(ascending=False)

featurs_top_var
# get ordered list features correlation with regression label RUL

data_final[features].corrwith(data_final.RUL).sort_values(ascending=False)
# get ordered list features correlation with regression label tot_RUL

data_final["Tot_RUL"]=data_final["Tot_RUL"].astype("int32")

data_final["tot_cycle"]=data_final["tot_cycle"].astype("int32")

data_final["cycle"]=data_final["cycle"].astype("int32")

data_final[features].corrwith(data_final.Tot_RUL).sort_values(ascending=False)
# plot a heatmap to display correlation:

import seaborn as sns

features.extend(["RUL","Tot_RUL","cycle","tot_cycle"])

cm = np.corrcoef(data_final[features].values.T)

sns.set(font_scale=1.0)

fig = plt.figure(figsize=(12, 10))

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 8}, yticklabels=features, xticklabels=features)

plt.title('Features Correlation Heatmap')

plt.show()