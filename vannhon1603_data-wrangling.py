import pandas as pd

import numpy as np

#dataset column names:



col_names = ['id','cycle','setting1','setting2','setting3','s1','s2','s3','s4','s5','s6','s7','s8','s9','s10','s11','s12','s13','s14','s15','s16','s17','s18','s19','s20','s21']

#load training data



df_train_raw = pd.read_csv('../input/PM_train1.txt', sep = ' ', header=None)

df_train_raw.head()

#drop extra space columnn



df_train_raw.drop([26,27], axis=1, inplace=True)
#assign column names



df_train_raw.columns = col_names

df_train_raw

# get some stat



df_train_raw.describe()
# check the data types



df_train_raw.dtypes
# check for NaN values



df_train_raw.isnull().sum()
#load test data



df_test_raw = pd.read_csv('../input/PM_test1.txt', sep = ' ', header=None)

df_test_raw.head()

#drop extra space columnn

df_test_raw.drop([26,27], axis=1, inplace=True)



#assign column names

df_test_raw.columns = col_names

df_test_raw.head()

# get some stat on test data



df_test_raw.describe()
# Load the truth data - actual 'ttf' for test data



df_truth = pd.read_csv('data/PM_truth.txt', sep = ' ', header=None)

df_truth.head()
#drop extra empty column in the truth data and rename remaining 'ttf'



df_truth.drop([1], axis=1, inplace=True)

df_truth.columns = ['ttf']

df_truth.head()
#get some stat on truth data



df_truth.describe()


def add_features(df_in, rolling_win_size):

    

    """Add rolling average and rolling standard deviation for sensors readings using fixed rolling window size.

    

    Args:

            df_in (dataframe)     : The input dataframe to be proccessed (training or test) 

            rolling_win_size (int): The window size, number of cycles for applying the rolling function

        

    Reurns:

            dataframe: contains the input dataframe with additional rolling mean and std for each sensor

    

    """

    

    sensor_cols = ['s1','s2','s3','s4','s5','s6','s7','s8','s9','s10','s11','s12','s13','s14','s15','s16','s17','s18','s19','s20','s21']

    

    sensor_av_cols = [nm.replace('s', 'av') for nm in sensor_cols]

    sensor_sd_cols = [nm.replace('s', 'sd') for nm in sensor_cols]

    

    df_out = pd.DataFrame()

    

    ws = rolling_win_size

    

    #calculate rolling stats for each engine id

    

    for m_id in pd.unique(df_in.id):

    

        # get a subset for each engine sensors

        df_engine = df_in[df_in['id'] == m_id]

        df_sub = df_engine[sensor_cols]



    

        # get rolling mean for the subset

        av = df_sub.rolling(ws, min_periods=1).mean()

        av.columns = sensor_av_cols

    

        # get the rolling standard deviation for the subset

        sd = df_sub.rolling(ws, min_periods=1).std().fillna(0)

        sd.columns = sensor_sd_cols

    

        # combine the two new subset dataframes columns to the engine subset

        new_ftrs = pd.concat([df_engine,av,sd], axis=1)

    

        # add the new features rows to the output dataframe

        df_out = pd.concat([df_out,new_ftrs])

        

    return df_out


def prepare_train_data (df_in, period):

    

    """Add regression and classification labels to the training data.



        Regression label: ttf (time-to-failure) = each cycle# for an engine subtracted from the last cycle# of the same engine

        Binary classification label: label_bnc = if ttf is <= parameter period then 1 else 0 (values = 0,1)

        Multi-class classification label: label_mcc = 2 if ttf <= 0.5* parameter period , 1 if ttf<= parameter period, else 2

        

      Args:

          df_in (dataframe): The input training data

          period (int)     : The number of cycles for TTF segmentation. Used to derive classification labels

          

      Returns:

          dataframe: The input dataframe with regression and classification labels added

          

    """

    

    #create regression label

    

    #make a dataframe to hold the last cycle for each enginge in the dataset

    df_max_cycle = pd.DataFrame(df_in.groupby('id')['cycle'].max())

    df_max_cycle.reset_index(level=0, inplace=True)

    df_max_cycle.columns = ['id', 'last_cycle']

    

    #add time-to-failure ttf as a new column - regression label

    df_in = pd.merge(df_in, df_max_cycle, on='id')

    df_in['ttf'] = df_in['last_cycle'] - df_in['cycle']

    df_in.drop(['last_cycle'], axis=1, inplace=True)

    

    #create binary classification label

    df_in['label_bnc'] = df_in['ttf'].apply(lambda x: 1 if x <= period else 0)

    

    #create multi-class classification label

    df_in['label_mcc'] = df_in['ttf'].apply(lambda x: 2 if x <= period/2 else 1 if x <= period else 0)

    

    return df_in

    


def prepare_test_data(df_test_in, df_truth_in, period):

    

    """Add regression and classification labels to the test data.



        Regression label: ttf (time-to-failure) = extract the last cycle for each enginge and then merge the record with the truth data

        Binary classification label: label_bnc = if ttf is <= parameter period then 1 else 0 (values = 0,1)

        Multi-class classification label: label_mcc = 2 if ttf <= 0.5* parameter period , 1 if ttf<= parameter period, else 2

        

      Args:

          df_in (dataframe): The input training data

          period (int)     : The number of cycles for TTF segmentation. Used to derive classification labels

          

      Returns:

          dataframe: The input dataframe with regression and classification labels added

    



    

    """

    

    df_tst_last_cycle = pd.DataFrame(df_test_in.groupby('id')['cycle'].max())

    

    df_tst_last_cycle.reset_index(level=0, inplace=True)

    df_tst_last_cycle.columns = ['id', 'last_cycle']

    

    df_test_in = pd.merge(df_test_in, df_tst_last_cycle, on='id')





    df_test_in = df_test_in[df_test_in['cycle'] == df_test_in['last_cycle']]



    df_test_in.drop(['last_cycle'], axis=1, inplace=True)

    

    df_test_in.reset_index(drop=True, inplace=True)

    

    df_test_in = pd.concat([df_test_in, df_truth], axis=1)

    

    #create binary classification label

    df_test_in['label_bnc'] = df_test_in['ttf'].apply(lambda x: 1 if x <= period else 0)

    

    #create multi-class classification label

    df_test_in['label_mcc'] = df_test_in['ttf'].apply(lambda x: 2 if x <= period/2 else 1 if x <= period else 0)



    return df_test_in
# add extracted features to training data



df_train_fx = add_features(df_train_raw, 5)

df_train_fx.head()
#add labels to training data using period of 30 cycles for classification



df_train = prepare_train_data (df_train_fx, 30)

df_train.head()
df_train.dtypes
# save the training data to csv file for later use



df_train.to_csv('data/train.csv', index=False)
# add extracted features to test data



df_test_fx = add_features(df_test_raw, 5)

df_test_fx.head()
#add labels to test data using period of 30 cycles for classification



df_test = prepare_test_data(df_test_fx, df_truth, 30)

df_test.head()
df_test.dtypes
# save the test data to csv file for later use



df_test.to_csv('data/test.csv', index=False)