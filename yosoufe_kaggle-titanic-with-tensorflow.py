import tensorflow as tf

import pandas as pd

import numpy as np
train_path = '../input/train.csv'

test_path = '../input/test.csv'

test_out = 'test_out.csv'
def read_files(path):

    return pd.read_csv(path)
train_sample = read_files(train_path)

test_sample = read_files(test_path)
print("How many survived in training data set:")

train_sample.groupby('Survived')['PassengerId'].nunique()
train_sample.groupby('Cabin').nunique().shape
print(train_sample['Fare'].mean(), train_sample['Fare'].std(), train_sample['Age'].mean(), train_sample['Age'].std())
def make_df_ready (df):

    def oneHot_encoding(phrase, x):

        if (x == phrase):

            return 1

        else:

            return 0 

    

    

    df['isMale']=np.vectorize(oneHot_encoding)(df['Sex'], 'male')

    df['isFemale']=np.vectorize(oneHot_encoding)(df['Sex'], 'female')

    df['isCherbourg']=np.vectorize(oneHot_encoding)(df['Embarked'], 'C')

    df['isQueenstown']=np.vectorize(oneHot_encoding)(df['Embarked'], 'Q')

    df['isSouthampton']=np.vectorize(oneHot_encoding)(df['Embarked'], 'S')

    df = df.drop(['Sex','Embarked','PassengerId','Name','Ticket','Cabin',],axis=1)

    df = df.fillna(df.mean())

    return df
train_df = make_df_ready(train_sample)

#train_df
def normalize(x,mean=0, std =0, isTrainigData = True):

    if isTrainigData:

        mean = x.mean(axis=0)

        std = x.std(axis=0)

    return (x-mean)/std, mean, std

# truns the data to numpy

data = train_df.values

#shuffle

np.random.shuffle(data)



# normalise

x_nor,mean,std = normalize(data[:,1:])



# split data

rat = 0.95



x_train = data[:int(rat*data.shape[0]),1:]

y_train = data[:int(rat*data.shape[0]),0]

x_valid = data[int(rat*data.shape[0]):,1:]

y_valid = data[int(rat*data.shape[0]):,0]

print (x_train.shape, y_train.shape)

print (x_valid.shape, y_valid.shape)
n_features = x_train.shape[1]

n_classes = 2



x_placeholder = tf.placeholder(tf.float64,(None,n_features), name='input')

prob = tf.placeholder(tf.float64)



hidden = tf.layers.dense(x_placeholder,256,name='hidden_1',  activation=tf.nn.tanh,

                         kernel_initializer=tf.truncated_normal_initializer() )

#hidden = tf.layers.dense(hidden,1024, activation=tf.nn.relu,

#                         kernel_initializer=tf.truncated_normal_initializer() )

hidden = tf.layers.dropout(hidden,rate = prob)

hidden = tf.layers.dense(hidden,1024, activation=tf.nn.sigmoid, 

                         kernel_initializer=tf.truncated_normal_initializer() )

#hidden = tf.layers.dense(hidden,10, activation=tf.nn.relu, 

#                         kernel_initializer=tf.truncated_normal_initializer() )



logit = tf.layers.dense(hidden,2,name='logit', 

                         kernel_initializer=tf.truncated_normal_initializer() )



y_placeholder = tf.placeholder(tf.int32, (None),name='output')

one_hot_y = tf.one_hot(y_placeholder, n_classes,name='onehot_output')
rate = 0.00005

reg_constant = 0.01

EPOCHS = 500

BATCH_SIZE = int(len(x_train)/1)



regularization_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels = one_hot_y)

loss = tf.reduce_mean(cross_entropy) + reg_constant * sum(regularization_loss)



optimizer = tf.train.AdamOptimizer(learning_rate=rate)

training_op = optimizer.minimize(loss)

# if predicton is correct

pred = tf.argmax(logit, 1)

correct_prediction = tf.equal(tf.argmax(logit,1), tf.argmax(one_hot_y,1))

accuracy_ops = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



def evaluate(x,y,sess):

    global BATCH_SIZE

    num_examples = len(x)

    total_accuracy = 0

    #sess = tf.get_default_session()

    for offset in range(0, num_examples, BATCH_SIZE):

        batch_x, batch_y = x[offset:offset+BATCH_SIZE], y[offset:offset+BATCH_SIZE]

        accuracy = sess.run(accuracy_ops, 

                          feed_dict={x_placeholder: batch_x, y_placeholder:batch_y, prob:1.0})

        total_accuracy += (accuracy*len(batch_x))

    return total_accuracy/num_examples
# initilisation of graph

import matplotlib.pyplot as plt

from sklearn.utils import shuffle



epochs = []

list_train_accuracy = []

list_valid_accuracy = []

plt.ion()
sess = tf.Session()

sess.run(tf.global_variables_initializer())

num_examples = len(x_train)

print("Training ..")

print()

for i in range(EPOCHS):

    x_train_sh, y_train_sh = x_train, y_train#shuffle(x_train, y_train)

    for offset in range(0, num_examples, BATCH_SIZE):

        end = offset + BATCH_SIZE

        batch_x, batch_y = x_train_sh[offset:end], y_train_sh[offset:end]

        sess.run(training_op,feed_dict={x_placeholder:batch_x, y_placeholder:batch_y,prob:1.0})



    valid_acc = evaluate(x_valid, y_valid,sess)

    train_acc = evaluate(x_train, y_train,sess)

    list_train_accuracy.append(train_acc)

    list_valid_accuracy.append(valid_acc)

    print('\r',"EPOCH {} ...".format(i+1),

          "Validation Accuracy = {:.3f} ...".format(valid_acc),

          "Training Accuracy = {:.3f} ...".format(train_acc),end='')

    if len(epochs) == 0:

        epochs.append(0)

    else:

        epochs.append(epochs[-1]+1)

plt.plot(epochs,list_train_accuracy,'b-',epochs,list_valid_accuracy,'r-')

plt.show()
test_df = make_df_ready(test_sample)

#test_df
test_data = test_df.values

test_normalized_x,_ ,_ = normalize(test_data,mean=mean,std=std,isTrainigData=False)
#with tf.Session() as sess:

predicted_labels = sess.run(pred, feed_dict={x_placeholder: test_normalized_x,prob: 1.0})

print(predicted_labels)
d = {'PassengerId': test_sample['PassengerId'], 'Survived': predicted_labels}

prediction_df = pd.DataFrame(data=d)

prediction_df.to_csv(test_out, index=False)