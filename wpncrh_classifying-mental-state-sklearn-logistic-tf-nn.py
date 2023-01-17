from sklearn.model_selection import train_test_split

import tensorflow as tf

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
df = pd.read_csv("../input/EEG data.csv")

subs=pd.read_csv("../input/demographic info.csv")
df = pd.read_csv("../input/EEG data.csv")

subs=pd.read_csv("../input/demographic info.csv")



print(df.columns.values)

print(df.shape)

print(subs.columns.values)
df.rename(columns={'0.000000000000000000e+00':'subject ID'},inplace=True)



print(df.columns.values)
print(subs.head)
for col in list(subs.columns):

	if subs[col].dtype == 'object':

		dums=pd.get_dummies(subs[col])

		subs = pd.concat([dums,subs], axis=1, join='outer')

		subs = subs.drop(col, 1)





print(subs.head())
merged=df.merge(subs,on='subject ID')

print(merged.head())
# set aside training

seed=7

def set_aside_test_data(d):

	label=d.pop("0.000000000000000000e+00.3") # pop off labels to new group

	x_train,x_test,y_train,y_test = train_test_split(d,label,test_size=0.2,random_state=seed)

	return x_train,x_test,y_train,y_test

	

x_train,x_test,y_train,y_test = set_aside_test_data(merged)
def prep_test_data_forTF(x,y):

    x_test=x.values

    y_test=pd.get_dummies(y)

    y_test=y_test.values

    return x_test, y_test



x_test_TF, y_test_TF = prep_test_data_forTF(x_test,y_test)
print(x_train.shape)
batch_size=1000

def get_mini_batch(x,y):

	rows=np.random.choice(x.shape[0], batch_size)

	return x[rows], y[rows]
def prep_train_data_forTF(x,y):

    x_train=x.values

    y_train=pd.get_dummies(y)

    y_train=y_train.values

    return x_train, y_train



x_train_numpy, y_train_numpy = prep_train_data_forTF(x_train,y_train)
print(y_train_numpy.shape)
# start session

sess = tf.Session()

lr=.0001

def trainNN(x_train_numpy, y_train_numpy,x_test_TF,y_test_TF,number_trials,number_nodes):

	# there are 8 features

	# place holder for inputs. feed in later

	x = tf.placeholder(tf.float32, [None, x_train_numpy.shape[1]])

	# # # take features  to 10 nodes in hidden layer

	w1 = tf.Variable(tf.random_normal([x_train.shape[1], 2],stddev=.5,name='w1'))

	# # # add biases for each node

	b1 = tf.Variable(tf.zeros([2]))

	# # calculate activations 

	#hidden_output = tf.nn.softmax(tf.matmul(x, w1) + b1)

	#w2 = tf.Variable(tf.random_normal([number_nodes, y_train_numpy.shape[1]],stddev=.5,name='w2'))

	#b2 = tf.Variable(tf.zeros([y_train_numpy.shape[1]]))

	# # placeholder for correct values 

	y_ = tf.placeholder("float", [None,y_train_numpy.shape[1]])

	# # #implement model. these are predicted ys

	y = tf.nn.softmax(tf.matmul(x, w1) + b1)

	# loss and optimization 

	loss = tf.reduce_mean(tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(y, y_, name='xentropy')))

	opt = tf.train.GradientDescentOptimizer(learning_rate=lr)

	train_step = opt.minimize(loss, var_list=[w1,b1])

	# init all vars

	init = tf.global_variables_initializer()

	sess.run(init)

	ntrials = number_trials

	for i in range(ntrials):

	    # get mini batch

	    a,b=get_mini_batch(x_train_numpy,y_train_numpy)

	    # run train step, feeding arrays of 100 rows each time

	    _, cost =sess.run([train_step,loss], feed_dict={x: a, y_: b})

	    if i%100 ==0:

	    	print("epoch is {0} and cost is {1}".format(i,cost))

	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	print("test accuracy is {}".format(sess.run(accuracy, feed_dict={x: x_test_TF, y_: y_test_TF})))





trainNN(x_train_numpy,y_train_numpy,x_test_TF,y_test_TF,5000,5)
from sklearn import linear_model
log_model = linear_model.LogisticRegression()
log_model.fit(X = x_train, 

              y = y_train)
print(log_model.coef_)
# Make predictions

preds = log_model.predict_proba(X= x_train)

predsdf = pd.DataFrame(preds)



from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score



predsdf['result'] = [0 if x >.5 else 1 for x in predsdf[0]]





accuracy_score(y_train, predsdf['result'])
preds = log_model.predict_proba(X= x_test)

predsdf = pd.DataFrame(preds)



from sklearn.metrics import accuracy_score



predsdf['result'] = [0 if x >.5 else 1 for x in predsdf[0]]





accuracy_score(y_test, predsdf['result'])
import numpy

import xgboost

from sklearn.metrics import accuracy_score


model = xgboost.XGBClassifier()

model.fit(x_train, y_train)
print(model)
# make predictions for test data

y_pred = model.predict(x_test)

predictions = [round(value) for value in y_pred]
# evaluate predictions

accuracy = accuracy_score(y_test, predictions)

print("Accuracy: %.2f%%" % (accuracy * 100.0))