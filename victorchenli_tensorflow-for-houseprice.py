import tensorflow as tf

# pandas

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline
train_df = pd.read_csv("../input/train.csv")

test_df    = pd.read_csv("../input/test.csv")

test_validation = pd.read_csv("../input/sample_submission.csv")
test_df = test_df.join(test_validation.set_index("Id"),on="Id", how="left")
all_df = pd.concat([train_df,test_df], axis=0)
#---------------- help functions ------------------



#convert all str category to integer

def category_to_int(category):

    #print category

    return hash(category)%32+1

    

def encode_col(col):

    encoder = LabelBinarizer()

    encoder.fit(col)

    return encoder.transform(col)

    

#deal with NaN

def fill_by_col(col):

    if col.dtype == np.object:

        col.fillna("EmptyStuff",inplace=True)

    else:

        rand = generate_std_err_int(col);

        col[np.isnan(col)] = rand

      

def generate_std_err_int(col):

    average   = col.mean()

    std       = col.std()

    count = col.isnull().sum()

    #min = average - std

    #max = average + std

    #return np.random.randint(min, max, size = count)

    return np.full([count,1] ,average)



def normalize(df):

    return (df - df.mean(axis=0)) / df.std(axis=0)

    #data = df

    #data = [np.log(tt + 1) for tt in data]

    #return data

    
all_df.drop(['Alley','PoolQC','Fence','MiscFeature'],axis=1, inplace=True)
# function for processing data set

def encode_data(data_df):

    for i in range(data_df.shape[1]):

        col = data_df.ix[:,i]

        #fill NaN fields

        fill_by_col(col)

    return data_df
all_encode_data = encode_data(all_df)
all_encode_dummies = pd.get_dummies(all_encode_data)
all_encode_normalize = normalize(all_encode_dummies)
from sklearn.model_selection import train_test_split

X = all_encode_normalize.drop(['SalePrice'], axis=1)

y = all_encode_normalize['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
print(X_train.shape, X_test.shape,y_train.shape,y_test.shape)

print(type(X_train),type(X_test),type(y_train),type(y_test))
# using tensorflow MNN



# === We build the graph here!

houseprice_graph = tf.Graph()



with houseprice_graph.as_default():



    HIDDEN_SIZE = 200

    num_features = X_train.shape[1]

    

    # create the neural network model

    keep_prob = tf.placeholder(tf.float32)

    prev_loss = tf.Variable(0., trainable=False)

    

    # first layer

    input_layer = tf.placeholder(tf.float32, [None, num_features], name='input')

    W1 = tf.Variable(tf.random_normal([num_features, HIDDEN_SIZE], stddev=.01), name='W1')

    b1 = tf.Variable(tf.random_normal([HIDDEN_SIZE], stddev=.01), name='b1')

    h1_layer = tf.add(tf.matmul(input_layer, W1), b1)

    h1_layer = tf.nn.relu(h1_layer)

    h1_layer = tf.nn.dropout(h1_layer, keep_prob, name='h1')

    

    # second layer

    W2 = tf.Variable(tf.random_normal([HIDDEN_SIZE, HIDDEN_SIZE], stddev=.01), name='W2')

    b2 = tf.Variable(tf.random_normal([HIDDEN_SIZE], stddev=.01), name='b2')

    h2_layer = tf.matmul(h1_layer, W2) + b2

    h2_layer = tf.nn.relu(h2_layer)

    h2_layer = tf.nn.dropout(h2_layer, keep_prob, name='h2')

    

    # third layer, output layer

    W3 = tf.Variable(tf.random_normal([HIDDEN_SIZE, 1], stddev=.01), name='W3')

    b3 = tf.Variable(tf.random_normal([1], stddev=.01), name='b3')

    output_layer = tf.matmul(h2_layer, W3) + b3

    y = tf.placeholder(tf.float32, shape=[None, 1], name='y')

    

    # cost function, optimizer to global minimal

    loss = tf.squared_difference(output_layer, y)

    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

    reg_constant = 0.01  # Choose an appropriate one.

    loss = loss + reg_constant * sum(reg_losses)

    loss = tf.reduce_mean(loss)

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-6, beta1=.85, beta2=.9).minimize(loss)

    init = tf.global_variables_initializer()
def get_next_batch(x_data,y_data,batch_size):

    idx = np.random.randint(0, len(x_data), batch_size)

    train_batches_x = x_data[idx]

    train_batches_y = y_data[idx]

    return train_batches_x, train_batches_y
X_train_array = X_train.as_matrix()

y_train_array = np.expand_dims(y_train,1)

X_test_array = X_test.as_matrix()

y_test_array = np.expand_dims(y_test,1)
# We create our sessions

sess = tf.Session(graph=houseprice_graph)



# Make sure to run the initialization

sess.run(init)



NUM_EPOCHS = 200000

BATCH_SIZE = 300

train_loss = []

valid_loss = []



# get the next batch

for i in range(NUM_EPOCHS):

    x_batch, y_batch = get_next_batch(X_train_array,y_train_array,BATCH_SIZE)

    

    sess.run(optimizer, feed_dict={input_layer: x_batch, y: y_batch, keep_prob: .75})

    

    train_loss.append(sess.run(loss, feed_dict={input_layer: x_batch, y: y_batch, keep_prob: .75}))

    valid_loss.append(sess.run(loss, feed_dict={input_layer: X_test_array, y: y_test_array, keep_prob: 1.}))

    

    if i%1000 == 0:

        print("--------Epochs:{}--------".format(i))

        print("train error:", train_loss[i])

        print("valid error:", valid_loss[i])



#print train_error;

#print "validation error:", sess.run(error, feed_dict={x:test_inputs, y:test_outputs})

sess.close()
