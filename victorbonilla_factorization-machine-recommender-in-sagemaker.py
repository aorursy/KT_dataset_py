!pip install sagemaker
import numpy as np
import pandas as pd
import boto3, io
from scipy.sparse import lil_matrix

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

%matplotlib inline
import matplotlib.pyplot as plt 
import seaborn as sns
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
aws_bucket = user_secrets.get_secret("AWS-BUCKET")
aws_access_key_id = user_secrets.get_secret("aws_access_key_id")
aws_secret_access_key = user_secrets.get_secret("aws_secret_access_key")
aws_region = 'eu-central-1'
df = pd.read_csv('/kaggle/input/ecommerce-dataset/events.csv')
df.head()
df.dtypes
vals = df.groupby('event').count().timestamp
plt.pie(vals, labels=vals.index, autopct='%1.2f%%')
df.loc[df['transactionid'].notnull(), ['visitorid', 'itemid']]
df_views = df.loc[df['event'] == 'view']
df_non_views = df.loc[df['event'] != 'view']

view_events_size = df_views.shape[0]
non_view_events_size = df_non_views.shape[0]
print(f'Count of non views events is {non_view_events_size} and events {view_events_size}')
# Picking a random sample from the views of equal size to non views
index_chosen = np.random.choice(len(df_views), size=non_view_events_size, replace=False)
random_views_choice = df_views.values[index_chosen]

# Concatenating and getting the balanced data
balanced_data = np.concatenate([df_non_views.values, random_views_choice])
assert balanced_data.shape[0] == non_view_events_size * 2
print(balanced_data.shape)
print(balanced_data.size)
balanced_data
visitor_enc = LabelEncoder()
visitor_enc2 = LabelEncoder()
vis_enc = visitor_enc.fit_transform(balanced_data[:,1])
visitor_enc2.fit(vis_enc)
assert visitor_enc.classes_.size == np.unique(balanced_data[:,1]).size
assert visitor_enc.classes_.size == visitor_enc2.classes_.size
assert visitor_enc2.classes_[-1] == np.unique(balanced_data[:,1]).size - 1
def encode_visitorid(visitorids):
    if type(visitorids) == np.ndarray:
        return visitor_enc2.transform(visitor_enc.transform(visitorids))
    else:
        return visitor_enc2.transform(visitor_enc.transform(np.array([visitorids])))


print(balanced_data[:10,1])
encode_visitorid(balanced_data[:10,1])
item_enc = LabelEncoder()
item_enc2 = LabelEncoder()
it_enc = item_enc.fit_transform(balanced_data[:,3])
item_enc2.fit(it_enc)
assert item_enc.classes_.size == np.unique(balanced_data[:,3]).size
assert item_enc.classes_.size == item_enc2.classes_.size
assert item_enc2.classes_[-1] == np.unique(balanced_data[:,3]).size - 1
def encode_itemid(itemids):
    return item_enc2.transform(item_enc.transform(itemids))


print(balanced_data[:10,3])
encode_itemid(balanced_data[:10,3])
visitors = visitor_enc2.classes_.size
items = item_enc2.classes_.size
features = visitors + items
print(f'Count of visitors is {visitors}, items {items}, and features {features}')
data = np.copy(balanced_data)
# Encoding variables
print(data)
data[:,1] = encode_visitorid(balanced_data[:,1])
data[:,3] = encode_itemid(balanced_data[:,3])
print(data)
# Split train and test
train, test = train_test_split(data, train_size=0.7, shuffle=True)
print(train.shape)
print(test.shape)
def data_to_X_y(data, features):
    X = lil_matrix((data.shape[0], features)).astype('float32')
    y = []
    for index, row in enumerate(data):
        X[index, row[1]] = 1.
        X[index, visitors + row[3]] = 1.
        if row[2] == 'view':
            y.append(0.)
        else:
            y.append(1.)
            
    y = np.array(y).astype('float32')
    return X, y
X_train, y_train = data_to_X_y(train, features)
assert X_train.shape == (train.shape[0], features)
assert y_train.size == train.shape[0]
X_test, y_test = data_to_X_y(test, features)
assert X_test.shape == (test.shape[0], features)
assert y_test.size == test.shape[0]
print(X_train[:10])
print(y_train[:10])
prefix = 'retailrocket-reco'

train_key      = 'train.protobuf'
train_prefix   = '{}/{}'.format(prefix, 'train')

test_key       = 'test.protobuf'
test_prefix    = '{}/{}'.format(prefix, 'test')

output_prefix  = 's3://{}/{}/output'.format(aws_bucket, prefix)
#session = boto3.session.Session(aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key, region_name=aws_region)

def dataset_to_protobuf_to_s3(X, y, prefix, key):
    buf = io.BytesIO()
    smac.write_spmatrix_to_sparse_tensor(buf, X, y)
    buf.seek(0)
    obj = '{}/{}'.format(prefix, key)
    session.resource('s3').Bucket(aws_bucket).Object(obj).upload_fileobj(buf)
    return 's3://{}/{}'.format(aws_bucket,obj)
    
    
#train_path = dataset_to_protobuf_to_s3(X_train, y_train, train_prefix, train_key)
#test_path = dataset_to_protobuf_to_s3(X_test, y_test, test_prefix, test_key)
  
#print(train_path)
#print(test_path)
#print('Output: {}'.format(output_prefix))
def visitor_to_X(visitor_id):
    """
    Function to get the input data for the model when asking for recommendations for a visitor.
    The result is a sparse matrix with the visitor_id and all the existing item_id.
    Input:
    visitor_id: endoded numpy array with one existing visitor_id
    """
    items_data = balanced_data[:,3]
    enc_unique_items_id = encode_itemid(np.unique(items_data))
    X = lil_matrix((enc_unique_items_id.size, features)).astype('float32')
    for index, item_id in enumerate(enc_unique_items_id):
        X[index, visitor_id[0]] = 1.
        X[index, visitor_id[0] + item_id + 1] = 1. # + 1 as a quick fix when item_id is 0
    
    return X


def decode_itemid(itemids):
    return item_enc.inverse_transform(item_enc2.inverse_transform(itemids))


def prediction_to_items(input_data, prediction):
    positive_index_y = np.nonzero(prediction)[0][:3] # Select three items here due to excess of memory allocation in the Kaggle server
    positive_matrix_x = np.take(input_data.toarray(), positive_index_y, axis=0)
    positive_index_x = np.nonzero(positive_matrix_x)
    items_reco = [x - 1 for x in positive_index_x[1][1::2]] # x - 1 due to indexing in sparse matrix with the visitor_id
    return decode_itemid(np.array(items_reco))


def get_three_recommendations(visitor_id):
    encoded_visitor_id = encode_visitorid(visitor_id)
    input_data = visitor_to_X(encoded_visitor_id)
    #prediction = fm_predictor.predict(input_data)
    prediction = y_test[:np.unique(balanced_data[:,3]).size] # This line is for testing purposes
    items_predicted = prediction_to_items(input_data, prediction)
    return items_predicted

visitor_id = 287857 # Test id
#visitor_id = 599528
get_three_recommendations(visitor_id)
