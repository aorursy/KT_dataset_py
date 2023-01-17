import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy.sparse as sp
from scipy.sparse import vstack
from scipy import sparse
from scipy.sparse.linalg import spsolve

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
from sklearn.model_selection import train_test_split
events = pd.read_csv('../input/events.csv')
category_tree = pd.read_csv('../input/category_tree.csv')
items1 = pd.read_csv('../input/item_properties_part1.csv')
items2 = pd.read_csv('../input/item_properties_part2.csv')
items = pd.concat([items1, items2])
user_activity_count = dict()
for row in events.itertuples():
    if row.visitorid not in user_activity_count:
        user_activity_count[row.visitorid] = {'view':0 , 'addtocart':0, 'transaction':0};
    if row.event == 'addtocart':
        user_activity_count[row.visitorid]['addtocart'] += 1 
    elif row.event == 'transaction':
        user_activity_count[row.visitorid]['transaction'] += 1
    elif row.event == 'view':
        user_activity_count[row.visitorid]['view'] += 1 

d = pd.DataFrame(user_activity_count)
dataframe = d.transpose()
# Activity range
dataframe['activity'] = dataframe['view'] + dataframe['addtocart'] + dataframe['transaction']
# removing users with only a single view
cleaned_data = dataframe[dataframe['activity']!=1]
# all users contains the userids with more than 1 activity in the events (4lac)
all_users = set(cleaned_data.index.values)
all_items = set(events['itemid'])
# todo: we need to clear items which are only viewed once

visitorid_to_index_mapping  = {}
itemid_to_index_mapping  = {}
vid = 0
iid = 0
for row in events.itertuples():
    if row.visitorid in all_users and row.visitorid not in visitorid_to_index_mapping:
        visitorid_to_index_mapping[row.visitorid] = vid
        vid = vid + 1

    if row.itemid in all_items and row.itemid not in itemid_to_index_mapping:
        itemid_to_index_mapping[row.itemid] = iid
        iid = iid + 1
n_users = len(all_users)
n_items = len(all_items)
user_to_item_matrix = sp.dok_matrix((n_users, n_items), dtype=np.int8)
# We need to check whether we need to add the frequency of view, addtocart and transation.
# Currently we are only taking a single value for each row and column.
action_weights = [1,2,3]

for row in events.itertuples():
    if row.visitorid not in all_users:
        continue
    
    
    mapped_visitor_id = visitorid_to_index_mapping[row.visitorid]
    mapped_item_id    = itemid_to_index_mapping[row.itemid]
    
    value = 0
    if row.event == 'view':
        value = action_weights[0]
    elif row.event == 'addtocart':
        value = action_weights[1]        
    elif row.event == 'transaction':
        value = action_weights[2]
        
    current_value = user_to_item_matrix[mapped_visitor_id, mapped_item_id]
    if value>current_value:
        user_to_item_matrix[mapped_visitor_id, mapped_item_id] = value
        
user_to_item_matrix = user_to_item_matrix.tocsr()
user_to_item_matrix.shape
filtered_items = items[items.itemid.isin(all_items)]
# adding a fake property to filtered items, which do not have any property

fake_itemid = []
fake_timestamp = []
fake_property = []
fake_value = []
all_items_with_property = set(items.itemid)
for itx in list(all_items):
    if itx not in all_items_with_property:
        fake_itemid.insert(0, itx)
        fake_timestamp.insert(0, 0)
        fake_property.insert(0, 888)
        fake_value.insert(0, 0)
    
fake_property_dict = {'itemid':fake_itemid, 'timestamp':fake_timestamp, 'property':fake_property,
                     'value':fake_value}

fake_df = pd.DataFrame(fake_property_dict, columns=filtered_items.columns.values)
filtered_items = pd.concat([filtered_items, fake_df])
filtered_items['itemid'] = filtered_items['itemid'].apply(lambda x: itemid_to_index_mapping[x])
filtered_items = filtered_items.sort_values('timestamp', ascending=False).drop_duplicates(['itemid','property'])
filtered_items.sort_values(by='itemid', inplace=True)
item_to_property_matrix = filtered_items.pivot(index='itemid', columns='property', values='value')
item_to_property_matrix.shape
useful_cols = list()
cols = item_to_property_matrix.columns
for col in cols:
    value = len(item_to_property_matrix[col].value_counts())
    if value < 50:
        useful_cols.insert(0, col)
item_to_property_matrix = item_to_property_matrix[useful_cols]
#Replace everything with ones and zeros
item_to_property_matrix_one_hot_sparse = pd.get_dummies(item_to_property_matrix)
from lightfm import LightFM
import scipy.sparse as sp
from scipy.sparse import vstack
item_to_property_matrix_one_hot_sparse.shape
from scipy.sparse import csr_matrix
item_to_property_matrix_sparse = csr_matrix(item_to_property_matrix_one_hot_sparse.values)
import random
def make_train(ratings, pct_test = 0.2):
    '''
    This function will take in the original user-item matrix and "mask" a percentage of the original ratings where a
    user-item interaction has taken place for use as a test set. The test set will contain all of the original ratings, 
    while the training set replaces the specified percentage of them with a zero in the original ratings matrix. 
    
    parameters: 
    
    ratings - the original ratings matrix from which you want to generate a train/test set. Test is just a complete
    copy of the original set. This is in the form of a sparse csr_matrix. 
    
    pct_test - The percentage of user-item interactions where an interaction took place that you want to mask in the 
    training set for later comparison to the test set, which contains all of the original ratings. 
    
    returns:
    
    training_set - The altered version of the original data with a certain percentage of the user-item pairs 
    that originally had interaction set back to zero.
    
    test_set - A copy of the original ratings matrix, unaltered, so it can be used to see how the rank order 
    compares with the actual interactions.
    
    user_inds - From the randomly selected user-item indices, which user rows were altered in the training data.
    This will be necessary later when evaluating the performance via AUC.
    '''
    test_set = ratings.copy() # Make a copy of the original set to be the test set. 
    test_set[test_set != 0] = 1 # Store the test set as a binary preference matrix
    training_set = ratings.copy() # Make a copy of the original data we can alter as our training set. 
    nonzero_inds = training_set.nonzero() # Find the indices in the ratings data where an interaction exists
    nonzero_pairs = list(zip(nonzero_inds[0], nonzero_inds[1])) # Zip these pairs together of user,item index into list
    random.seed(0) # Set the random seed to zero for reproducibility
    num_samples = int(np.ceil(pct_test*len(nonzero_pairs))) # Round the number of samples needed to the nearest integer
    samples = random.sample(nonzero_pairs, num_samples) # Sample a random number of user-item pairs without replacement
    user_inds = [index[0] for index in samples] # Get the user row indices
    item_inds = [index[1] for index in samples] # Get the item column indices
    training_set[user_inds, item_inds] = 0 # Assign all of the randomly chosen user-item pairs to zero
    training_set.eliminate_zeros() # Get rid of zeros in sparse array storage after update to save space
    return training_set, test_set, list(set(user_inds)) # Output the unique list of user rows that were altered  
X_train, X_test, item_users_altered = make_train(user_to_item_matrix, pct_test = 0.1)
no_comp, lr, ep = 30, 0.01, 10 
model = LightFM(no_components=no_comp, learning_rate=lr, loss='warp')
model.fit_partial(
        X_train,
        item_features=item_to_property_matrix_sparse,
        epochs=ep,
        num_threads=4,
        verbose=True)

from sklearn import metrics
def auc_score(predictions, target):
    '''
    This simple function will output the area under the curve using sklearn's metrics. 
    
    parameters:
    - predictions: your prediction output
    - test: the actual target result you are comparing to
    returns:
    - AUC (area under the Receiver Operating Characterisic curve)
    '''
    fpr, tpr, thresholds = metrics.roc_curve(target, predictions)
    return metrics.auc(fpr, tpr)

def normalise_for_predictions(arr):
    arr[arr <= 1.5] = 0
    arr[arr > 1.5] = 1
    return arr

def get_predictions(user_id, model):
    pid_array = np.arange(n_items, dtype=np.int32)
    uid_array = np.empty(n_items, dtype=np.int32)
    uid_array.fill(user_id)
    predictions = model.predict(
            uid_array,
            pid_array,
            item_features=item_to_property_matrix_sparse,
            num_threads=4)
    
    return predictions
    
def calc_mean_auc(training_set, altered_users, model, test_set):
    '''
    This function will calculate the mean AUC by user for any user that had their user-item matrix altered. 
    
    parameters:
    
    training_set - The training set resulting from make_train, where a certain percentage of the original
    user/item interactions are reset to zero to hide them from the model 
    
    predictions - The matrix of your predicted ratings for each user/item pair as output from the implicit MF.
    These should be stored in a list, with user vectors as item zero and item vectors as item one. 
    
    altered_users - The indices of the users where at least one user/item pair was altered from make_train function
    
    test_set - The test set constucted earlier from make_train function
    
    
    
    returns:
    
    The mean AUC (area under the Receiver Operator Characteristic curve) of the test set only on user-item interactions
    there were originally zero to test ranking ability in addition to the most popular items as a benchmark.
    '''
    
    
    store_auc = [] # An empty list to store the AUC for each user that had an item removed from the training set
    popularity_auc = [] # To store popular AUC scores
    pop_items = np.array(test_set.sum(axis = 0)).reshape(-1) # Get sum of item iteractions to find most popular
    print (len(altered_users))
    index = 0;
    for user in altered_users: # Iterate through each user that had an item altered
        print (index)
        index = index + 1
        training_row = training_set[user,:].toarray().reshape(-1) # Get the training set row
        zero_inds = np.where(training_row == 0) # Find where the interaction had not yet occurred
        # Get the predicted values based on our user/item vectors
        pred = get_predictions(user, model)[zero_inds].reshape(-1)
        pred = normalise_for_predictions(pred)
        # Get only the items that were originally zero
        # Select all ratings from the MF prediction for this user that originally had no iteraction
        actual = test_set[user,:].toarray()[0,zero_inds].reshape(-1) 
        actual = normalise_for_predictions(actual)
        # Select the binarized yes/no interaction pairs from the original full data
        # that align with the same pairs in training 
        pop = pop_items[zero_inds] # Get the item popularity for our chosen items
        store_auc.append(auc_score(pred, actual)) # Calculate AUC for the given user and store
        popularity_auc.append(auc_score(pop, actual)) # Calculate AUC using most popular and score
    # End users iteration
    return float('%.3f'%np.mean(store_auc)), float('%.3f'%np.mean(popularity_auc))  
   # Return the mean AUC rounded to three decimal places for both test and popularity benchmark
    


calc_mean_auc(X_train, item_users_altered,  model, X_test)
predictions