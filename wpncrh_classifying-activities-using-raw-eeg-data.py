import json

import random

import pandas as pd

import tensorflow as tf

import numpy as np

from sklearn.model_selection import train_test_split



df = pd.read_csv("../input/eeg-data.csv")

subs=pd.read_csv("../input/subject-metadata.csv")
print(subs.head())
# rename ID to id so i can match to eeg dataframe

subs.rename(columns={'ID':'id'},inplace=True)



# convert categorical data to dummies

for col in list(subs.columns):

	if subs[col].dtype == 'object':

		dums=pd.get_dummies(subs[col])

		subs = pd.concat([dums,subs], axis=1, join='outer')

		subs = subs.drop(col, 1)





print(subs.head())


def prepare_individual_data(df,individual):

	# drop unused features. just leave eeg_power and the label

	df = df.drop('Unnamed: 0', 1)

	df = df.drop('indra_time', 1)

	df = df.drop('browser_latency', 1)

	df = df.drop('reading_time', 1)

	df = df.drop('attention_esense', 1)

	df = df.drop('meditation_esense', 1)

	df = df.drop('signal_quality', 1)

	df = df.drop('createdAt', 1)

	df = df.drop('updatedAt', 1)

	df = df.drop('eeg_power',1)

	df['raw_values'] = df.raw_values.map(json.loads) #must perform, or else we won't be able to split cell by commas 

	# separate eeg power to multiple columns

	to_series = pd.Series(df['raw_values']) # df to series

	raw_data=pd.DataFrame(to_series.tolist()) #series to list and then back to df

	df = pd.concat([df,raw_data], axis=1, join='outer') # concatenate the create columns

	df = df.drop('raw_values', 1) # drop comma separated cell

	df=df.loc[df['id'] == individual]

	return df
individual_data=prepare_individual_data(df,1)

print(individual_data.shape)

print(individual_data.head())

# now we have all raw values for id 1. and labels
def clean_labels(dd):

    #Thanks Alexandru

	dd["label"] = dd["label"].str.replace("\d|Instruction|-|ver", "")

	return dd



cleaned_individual_data = clean_labels(individual_data)
def drop_useless_labels(df):

	# drop unlabeled and everyone paired.

	df = df[df.label != 'unlabeled']

	df = df[df.label != 'everyone paired']

	return df



final_individual_full_data= drop_useless_labels(cleaned_individual_data)
print(final_individual_full_data['label'].value_counts())
merged=final_individual_full_data.merge(subs,on='id')

print(merged.head())
# set aside training

def set_aside_test_data(d):

	label=d.pop("label") # pop off labels to new group

	x_train,x_test,y_train,y_test = train_test_split(d,label,test_size=0.2)

	return x_train,x_test,y_train,y_test

	

x_train,x_test,y_train,y_test = set_aside_test_data(merged)
def prep_test_data(x,y):

    x_test=x.values

    y_test=pd.get_dummies(y)

    y_test=y_test.values

    return x_test, y_test



x_test, y_test = prep_test_data(x_test,y_test)

x_test, y_test =  np.array(x_test,dtype='float32'), np.array(y_test,dtype='float32')
print(x_train.shape)

print(x_train.head())
def combine_and_grow_training_data(x,y,ls):

    full_train=pd.concat([x, y], axis=1)

    loop_size=ls

    for i in range(loop_size):

        copy = full_train

        copy[1]=copy[1]+random.gauss(1,.1) # add noise to mean freq var

        full_train=full_train.append(copy,ignore_index=True) # make voice df 2x as big

        print("shape of df after {0}th iteration of this loop is {1}".format(i,full_train.shape))

    return full_train



full_train = combine_and_grow_training_data(x_train,y_train,10)
print(full_train.columns.values)
def prepare_for_training(f):

    train_labels=f.pop("label") # pop off labels to new group

    train_labels=pd.get_dummies(train_labels) # convert labels to one hot

    train_labels=train_labels.values

    features = full_train.values

    x_train,y_train = np.array(features,dtype='float32'), np.array(train_labels,dtype='float32')

    return x_train, y_train



x_train, y_train = prepare_for_training(full_train)
batch_size=100

def get_mini_batch(x,y):

	rows=np.random.choice(x.shape[0], batch_size)

	return x[rows], y[rows]
print(y_train.shape)
# start session

sess = tf.Session()

    

def trainNN(x_train, y_train,x_test,y_test,number_trials,number_nodes):

	# there are 8 features

	# place holder for inputs. feed in later

	x = tf.placeholder(tf.float32, [None, x_train.shape[1]])

	# # # take 20 features  to 10 nodes in hidden layer

	w1 = tf.Variable(tf.random_normal([x_train.shape[1], number_nodes],stddev=.5,name='w1'))

	# # # add biases for each node

	b1 = tf.Variable(tf.zeros([number_nodes]))

	# # calculate activations 

	hidden_output = tf.nn.softmax(tf.matmul(x, w1) + b1)

	w2 = tf.Variable(tf.random_normal([number_nodes, y_train.shape[1]],stddev=.5,name='w2'))

	b2 = tf.Variable(tf.zeros([y_train.shape[1]]))

	# # placeholder for correct values 

	y_ = tf.placeholder("float", [None,y_train.shape[1]])

	# # #implement model. these are predicted ys

	y = tf.nn.softmax(tf.matmul(hidden_output, w2) + b2)

	# loss and optimization 

	loss = tf.reduce_mean(tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(y, y_, name='xentropy')))

	opt = tf.train.AdamOptimizer(learning_rate=.0001)

	train_step = opt.minimize(loss, var_list=[w1,b1,w2,b2])

	# init all vars

	init = tf.initialize_all_variables()

	sess.run(init)

	ntrials = number_trials

	for i in range(ntrials):

	    # get mini batch

	    a,b=get_mini_batch(x_train,y_train)

	    # run train step, feeding arrays of 100 rows each time

	    _, cost =sess.run([train_step,loss], feed_dict={x: a, y_: b})

	    if i%500 ==0:

	    	print("epoch is {0} and cost is {1}".format(i,cost))

	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	print("test accuracy is {}".format(sess.run(accuracy, feed_dict={x: x_test, y_: y_test})))





trainNN(x_train,y_train,x_test,y_test,10000,1000)
sess.close()
def prepare_individual_data_with_EEGBands(df,individual):

	# drop unused features. just leave eeg_power and the label

	df = df.drop('Unnamed: 0', 1)

	df = df.drop('indra_time', 1)

	df = df.drop('browser_latency', 1)

	df = df.drop('reading_time', 1)

	df = df.drop('attention_esense', 1)

	df = df.drop('meditation_esense', 1)

	df = df.drop('signal_quality', 1)

	df = df.drop('createdAt', 1)

	df = df.drop('updatedAt', 1)

	df['raw_values'] = df.raw_values.map(json.loads) 

	df['eeg_power'] = df.eeg_power.map(json.loads) 

	eeg_to_series = pd.Series(df['eeg_power'])

	eeg_raw_data=pd.DataFrame(eeg_to_series.tolist())

	df = pd.concat([df,eeg_raw_data], axis=1, join='outer') # concatenate the create columns

	df = df.drop('eeg_power', 1) # drop comma separated cell

	raw_to_series = pd.Series(df['raw_values']) # df to series

	raw_data=pd.DataFrame(raw_to_series.tolist()) #series to list and then back to df

	df = pd.concat([df,raw_data], axis=1, join='outer') 

	df = df.drop('raw_values', 1) # drop comma separated cell

	df=df.loc[df['id'] == individual]

	return df
individual_data=prepare_individual_data_with_EEGBands(df,1)



print(individual_data.head())
cleaned_individual_data = clean_labels(individual_data)

final_individual_full_data= drop_useless_labels(cleaned_individual_data)

merged=final_individual_full_data.merge(subs,on='id')

x_train,x_test,y_train,y_test = set_aside_test_data(final_individual_full_data)

x_test, y_test = prep_test_data(x_test,y_test)

x_test, y_test =  np.array(x_test,dtype='float32'), np.array(y_test,dtype='float32')  
full_train = combine_and_grow_training_data(x_train,y_train,10)

x_train, y_train = prepare_for_training(full_train)
# start session

sess = tf.Session()
trainNN(x_train,y_train,x_test,y_test,10000,1000)
sess.close()