#Load the required packages
import numpy as np
import pandas as pd
import time
import psutil
import multiprocessing as mp

#check the number of cores
num_cores = mp.cpu_count()
print("This kernel has ",num_cores,"cores and you can find the information regarding the memory usage:",psutil.virtual_memory())
# Writing as a function
def process_user_log(chunk):
    grouped_object = chunk.groupby(chunk.index,sort = False) # not sorting results in a minor speedup
    func = {'date':['min','max','count'],'num_25':['sum'],'num_50':['sum'], 
            'num_75':['sum'],'num_985':['sum'],
           'num_100':['sum'],'num_unq':['sum'],'total_secs':['sum']}
    answer = grouped_object.agg(func)
    return answer
# Number of rows
size = 4e7 # 40 millions
reader = pd.read_csv('../input/user_logs.csv', chunksize = size, index_col=['msno'])
start_time = time.time()

for i in range(10):
    user_log_chunk = next(reader)
    if(i==0):
        result = process_user_log(user_log_chunk)
        print("Number of rows ",result.shape[0])
        print("Loop ",i,"took %s seconds" % (time.time() - start_time))
    else:
        result = result.append(process_user_log(user_log_chunk))
        print("Number of rows ",result.shape[0])
        print("Loop ",i,"took %s seconds" % (time.time() - start_time))
    del(user_log_chunk)    

# Unique users vs Number of rows after the first computation    
print(len(result))
check = result.index.unique()
print(len(check))

result.columns = ['_'.join(col).strip() for col in result.columns.values]    
func = {'date_min':['min'],'date_max':['max'],'date_count':['count'] ,
           'num_25_sum':['sum'],'num_50_sum':['sum'],
           'num_75_sum':['sum'],'num_985_sum':['sum'],
           'num_100_sum':['sum'],'num_unq_sum':['sum'],'total_secs_sum':['sum']}
processed_user_log = result.groupby(result.index).agg(func)
print(len(processed_user_log))
processed_user_log.columns = processed_user_log.columns.get_level_values(0)
processed_user_log.head()
processed_user_log.info(), processed_user_log.describe()
processed_user_log = processed_user_log.reset_index(drop = False)

# Initialize the dataframes dictonary
dict_dfs = {}

# Read the csvs into the dictonary
dict_dfs['processed_user_log'] = processed_user_log

def get_memory_usage_datafame():
    "Returns a dataframe with the memory usage of each dataframe."
    
    # Dataframe to store the memory usage
    df_memory_usage = pd.DataFrame(columns=['DataFrame','Memory MB'])

    # For each dataframe
    for key, value in dict_dfs.items():
    
        # Get the memory usage of the dataframe
        mem_usage = value.memory_usage(index=True).sum()
        mem_usage = mem_usage / 1024**2
    
        # Append the memory usage to the result dataframe
        df_memory_usage = df_memory_usage.append({'DataFrame': key, 'Memory MB': mem_usage}, ignore_index = True)
    
    # return the dataframe
    return df_memory_usage

init = get_memory_usage_datafame()

dict_dfs['processed_user_log']['date_min'] = dict_dfs['processed_user_log']['date_min'].astype(np.int32)
dict_dfs['processed_user_log']['date_max'] = dict_dfs['processed_user_log'].date_max.astype(np.int32)
dict_dfs['processed_user_log']['date_count'] = dict_dfs['processed_user_log']['date_count'].astype(np.int8)
dict_dfs['processed_user_log']['num_25_sum'] = dict_dfs['processed_user_log'].num_25_sum.astype(np.int32)
dict_dfs['processed_user_log']['num_50_sum'] = dict_dfs['processed_user_log'].num_50_sum.astype(np.int32)
dict_dfs['processed_user_log']['num_75_sum'] = dict_dfs['processed_user_log'].num_75_sum.astype(np.int32)
dict_dfs['processed_user_log']['num_985_sum'] = dict_dfs['processed_user_log'].num_985_sum.astype(np.int32)
dict_dfs['processed_user_log']['num_100_sum'] = dict_dfs['processed_user_log'].num_100_sum.astype(np.int32)
dict_dfs['processed_user_log']['num_unq_sum'] = dict_dfs['processed_user_log'].num_unq_sum.astype(np.int32)

init.join(get_memory_usage_datafame(), rsuffix = '_managed')
import matplotlib.pyplot as plt

data = init.join(get_memory_usage_datafame(), rsuffix = '_managed')
plt.style.use('ggplot')
data.plot(kind='bar',figsize=(10,10), title='Memory usage');
