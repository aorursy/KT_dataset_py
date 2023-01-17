import numpy as np

import pandas as pd

from dask_ml.preprocessing import StandardScaler

import gc

import time

import dask.dataframe as dask

from dask.distributed import Client, progress 
# set workers

client = Client(n_workers=2, threads_per_worker=2, memory_limit='2GB')

client # work locally only 
# setting the number of rows for the CSV file

start_time = time.time()



N = 5_000_000

columns = 257



# create DF 

df = pd.DataFrame(np.random.randint(999, 999999, size=(N, columns)), columns=['level_%s' % i for i in range(0, columns)])



print('%s seconds' % (time.time() - start_time))
display(df.head(2))
print(f'shape of generated data is {df.shape}')
# # save df to csv 



# start_time = time.time()



# df.to_csv('random.csv', sep=',')



# print('%s seconds' % (time.time() - start_time)) # 877.5422155857086 seconds, 8.9 G
test = '../input/test.tsv'

train = '../input/train.tsv'
class LoadBigCsvFile:

    

    '''load data from tsv, transform, scale, add two columns

    Input .csv, .tsv files

    Output transformed file ready to save in .csv, .tsv format

    '''

    def __init__(self, train, test, scaler=StandardScaler(copy=False)):



        self.train = train

        self.test = test

        self.scaler = scaler # here we use StandartScaler of Dask. We can use sklearn one



    def read_data(self):



        # use dask and load with smallest possible format - int16 using 'C'

        try:

            data_train = dask.read_csv(self.train, \

                                     dtype={n:'int16' for n in range(1, 300)}, engine='c').reset_index()

            data_test = dask.read_csv(self.test, \

                                    dtype={n:'int16' for n in range(1, 300)}, engine='c').reset_index()

        except: (IOError, OSError), 'can not open file'



        #if any data?

        assert len(data_test) != 0 and len(data_train) != 0, 'No data in files'



        # fit train and transform test

        self.scaler.fit(data_train.iloc[:,1:])

        del data_train # del file that we do not need

        test_transformed = self.scaler.transform(data_test.iloc[:,1:])



        # compute  values and add columns

        test_transformed['max_feature_2_abs_mean_diff'] = abs(test_transformed.mean(axis=1) - test_transformed.max(axis=1))

        test_transformed['max_feature_2_index'] = test_transformed.idxmin(axis=1)

        test_transformed['job_id'] = data_test.iloc[:,0] # add first column (it is not numerical)



        del data_test # del file that we do not need



        return test_transformed
start_time = time.time()

data = LoadBigCsvFile(train, test).read_data()

gc.collect()

print('class loaded in %s seconds' % (time.time() - start_time))
# save to hdf for later use or modification

start_time = time.time()

data.to_hdf('test_proc.hdf',  key='df1')

print('file saved in hdf in %s seconds' % (time.time() - start_time))
start_time = time.time()

hdf_read = dask.read_hdf('test_proc.hdf', key='df1', mode='r', chunksize=10000)

print('file load into system in %s seconds' % (time.time() - start_time))
display(hdf_read.head(3))