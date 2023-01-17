__author__ = "Victor Xu"

__email__ = "victor.c.xu@gmail.com"

__website__ = "victorxu.me"



__powerpoint_presentation__ = "https://www.dropbox.com/s/ezenffm4bbrutar/grocer.pdf?dl=0"

__write_up__ = "coming soon"



__copyright__ = "Copyright 2019, Victor Xu"
import time

import os

import random



import pandas as pd

import numpy as np



#from polish_grocer_utility_script import load_monthly_data

from sklearn.preprocessing import MinMaxScaler

from scipy.spatial.distance import cosine

from multiprocessing import Process, Queue, Lock
def load_monthly_data():

    """Loads monthly sales data from  csv"""



    monthly = pd.read_csv(

        '/kaggle/input/total-sale-2018-yearly-data-of-grocery-shop/SELL_1.csv',

        delimiter=';',

        decimal=',',

        encoding='latin-1')

    monthly = monthly.rename(columns={

        'Date': 'date',

        'PKod': 'sku',

        'Pgroup': 'group',

        'Pname': 'name',

        'Pquantity': 'quantity',

        'pce_zn': 'unit_cogs',

        'pwa_zn': 'agg_cogs',  # aggregate cogs for this sku

        'pce_sn': 'unit_revenue',

        'pwn_sn': 'agg_rev',  # aggregate revenue for this sku

        'pmarza': 'gross_margin',

        'pmarzajedn': 'unit_contribution',

        'pkwmarza': 'agg_dollar_contribution',

        'pudzmarza': 'share_of_margin'

    })



    monthly = monthly.drop(['pwa_sn', 'pce_sb', 'pwa_sb', 'pudzsb'], axis=1)

    monthly.group = monthly.group.str.lower()

    monthly.name = monthly.name.str.lower()

    monthly.date = pd.to_datetime(monthly.date, format="%d.%m.%Y")

    monthly.unit_cogs = monthly.unit_cogs.str.replace(

        ',', '.').str.replace(

        ' ', '')



    monthly.group = monthly.group.replace(

        {"ketch_concetrate_mustard_majo_horseradish": "sauce"})



    return monthly
monthly = load_monthly_data()
monthly.sample(3)
# Build product information df

product_description_df = monthly[['sku','group','name']].set_index('sku', drop=True)

product_description_df = product_description_df.drop_duplicates()

product_description_df.head()
utility_matrix = monthly.pivot_table(values='quantity', index='sku', columns='date', aggfunc='sum')



# Replace NaN with 0s, because NaN indicates no sale for that month

utility_matrix = utility_matrix.fillna(0)
utility_matrix.sample(5)
um_norm = utility_matrix.sub(utility_matrix.mean(axis=1), axis=0)



# MinMaxScaler can only scale column wise, we need to scale row wise

# have to transpose and then transpose it back

um_norm = np.transpose(um_norm.values)

um_norm = MinMaxScaler(feature_range=(0,1)).fit_transform(um_norm)

um_norm = np.transpose(um_norm)

um_norm_df = pd.DataFrame(um_norm, index=utility_matrix.index)
def producer(queue, lock, um_norm_df):

    """

    Iterate through the normalized utility matrix DataFrame and

    adds all the pairs of vectors that need the cosine similarity to

    be calculated

    """

    

    # Acquire lock on console

    with lock:

        print('Producer {} is starting.'.format(os.getpid()))

    

    counter = 0

    batch = []

    # put data required to calculate item-item cosine similarity in queue

    for idx_i, row_data_i in um_norm_df.iterrows():

        for idx_j, row_data_j in um_norm_df.iterrows():

            # skip duplicate computations. w[i,j] and w[j,i] are the same

            if idx_i < idx_j:

                data = (idx_i, idx_j, row_data_i.values, row_data_j.values)

                batch.append(data)

                

                # Queue data every 5000 iterations to minimize consumer I/O

                if counter % 5000 == 0:

                    queue.put(batch)

                    batch = []

                

                counter += 1

    

    # Put left over data in queue

    queue.put(batch)  

    

    # Acquire lock on console

    with lock:

        print('Producer {} is finished. Quitting.'.format(os.getpid()))

 

 

def consumer(task_queue, lock, result_queue):

    """

    Get data from the task queue and calculates the cosine similarity.

    

    Note: Item similarity with itself will not be calculated and as such,

    will have a value of 0 instead of 1 in the final W matrix (item-item matrix).

    This property is desirable because we never want to recommend the same item

    itself.

    """



    with lock:

        print('Consumer {} is starting.'.format(os.getpid()))

     

    while True:

        # Get data, if queue is empty, it will block until producer 

        # put data in queue 

        batch = task_queue.get()

        result_batch= []

        

        # Iterate over the batch of data, each is a vector pair

        for data in batch:

            idx_i, idx_j, row_data_i, row_data_j = data



            # scipy cosine calculates the cosine distance. 1 - distance = similarity

            similarity = 1. - cosine(row_data_i, row_data_j)

            

            result_batch.append((idx_i, idx_j, similarity))

            result_batch.append((idx_j, idx_i, similarity))

        

        result_queue.put(result_batch)



        

        

    with lock:

        print('Consumer {} is finished. Quiting'.format(os.getpid()))





def process_result_queue(result_queue, item_item_df):

    """

    Process the return data in result queue and update the item-item DataFrame

    """

    

    while result_queue.qsize() > 0:

        result_batch = result_queue.get()

        

        for result in result_batch:

            idx_i, idx_j, similarity = result



            item_item_df.loc[idx_i,idx_j] = similarity

    

    return item_item_df

    

    

    

def build_item_item_matrix(um_norm_df):

    """Calculates the item-item similarity matrix"""

    

    num_skus = um_norm.shape[0]



    item_item_df = pd.DataFrame(np.zeros((num_skus, num_skus)),

                                index=utility_matrix.index, 

                                columns=utility_matrix.index)

    

    # For testing

    # um_norm_df =  um_norm_df.iloc[:4,:4].copy()

    

    task_queue = Queue()

    result_queue = Queue()



    # Create a lock object to synchronize resource access

    lock = Lock()



    producers = []

    consumers = []



    # Create producer processes

    producers.append(Process(target=producer, args=(task_queue, lock, um_norm_df)))



    # Create consumer processes

    for i in range(8):

        p = Process(target=consumer, args=(task_queue, lock, result_queue))



        # Set daemon to true so consumers will exit, otherwise it will be in inf loop

        p.daemon = True

        consumers.append(p)



    for p in producers:

        p.start()



    for c in consumers:

        c.start()



    # Like threading, we have a join() method that synchronizes our program

    for p in producers:

        p.join()



    item_item_df = process_result_queue(result_queue, item_item_df)



    print('All done')

    

    return item_item_df
def get_n_similar_items_for_sku(sku, n, item_item_df, 

                                product_description_df, print_item_desc=False,

                               return_as_json=False):

    """

    Looks through the item-item similarity matrix and returns the 

    top n similar items for a given sku

    

    Args:

        sku: int

            SKU to get top n similar items

        

        n: int

            Number of similar items to return

        

        item_item_df: DataFrame

            Pre-computed item-item similarity matrix as pd.DataFrame

            and indexed on integer valued SKUs

            

        product_description_df: DataFrame

            Df holding item descriptions indexed on integer valued SKUs

            

        print_item_desc: bool

            Set true to print requestd sku product name and category

        

        return_as_json: bool

            Returns JSON instead of DataFrame

    

    Returns:

        items: DataFrame or JSON

            Df or JSON containing top n similar items with product information

            Use return_as_json argument to set return data type

    """

    

    top_n_skus = item_item_df.loc[:,sku].sort_values(ascending=False).iloc[:n]

    top_n_skus = pd.DataFrame(top_n_skus).join(product_description_df, how='left')   

    top_n_skus.columns = ['similarity','category','name']

    

    if print_item_desc:

        item = product_description_df.loc[sku]

        s = "Top {} items /w similar purchase pattern to {} ({}) are as follows"

        print(s.format(n, item.loc["name"], item.group))

    

    if return_as_json:

        return top_n_skus.to_json(orient="records")

    

    return top_n_skus
# Build the item-item similarity matrix, aka W matrix

item_item_df = build_item_item_matrix(um_norm_df)
get_n_similar_items_for_sku(304, 15, item_item_df, 

                            product_description_df, 

                            print_item_desc=True)
get_n_similar_items_for_sku(9664, 15, item_item_df, 

                            product_description_df, 

                            print_item_desc=True)
get_n_similar_items_for_sku(304, 2, item_item_df, 

                            product_description_df, 

                            print_item_desc=False,

                            return_as_json=True)
#import pickle



#with open('../model/item_item_df.pickle','wb') as f:

#    pickle.dump((item_item_df, product_description_df), f)
print("Done")