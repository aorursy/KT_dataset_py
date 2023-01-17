import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
train = pd.read_csv('../input/train.csv')
train_data = train.drop(["label"],axis=1)
train_label = pd.get_dummies(train["label"],columns=np.arange(9))
digit = train_data.iloc[10]
print(train_label.iloc[10])
plt.imshow(digit.values.reshape((28,28)),cmap="Greys")
plt.show()
X_train, X_test, y_train, y_test = train_test_split(train_data.values,train_label.values, random_state=13,test_size=0.2)
sess = tf.InteractiveSession()
train_size,num_features = X_train.shape
print(X_train.shape)
num_labels = y_train.shape[1]
print(num_labels)
batch_size=100
n_hidden1 = 500
n_hidden2 = 500
n_outputs = num_labels
x = tf.placeholder(dtype=tf.float32,shape=[None, num_features], name="x")
y_ = tf.placeholder(dtype=tf.int64,shape=[None, num_labels], name="labels")

with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(x,n_hidden1,name="hidden1",activation=tf.nn.relu)
    hidden2 = tf.layers.dense(hidden1,n_hidden2,name="hidden2",activation=tf.nn.relu)
    logits = tf.layers.dense(hidden2,n_outputs,name="outputs")

with tf.name_scope("loss"):
    xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_,logits=logits)
    loss = tf.reduce_mean(xentropy)

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(0.001)
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    ypred = tf.nn.softmax(logits)
    correct_prediction = tf.equal(tf.argmax(ypred,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))    

init = tf.global_variables_initializer()

num_epochs = 10

sess.run(init)
for epoch in range(num_epochs):
        
    for step in range(int(train_size / batch_size)):
        batch_data = X_train[step*batch_size:step*batch_size+batch_size]
        batch_labels = y_train[step*batch_size:step*batch_size+batch_size]
        train = sess.run(training_op, feed_dict={x: batch_data, y_:batch_labels})
        
            
    acc = sess.run(accuracy,feed_dict={x:X_test, y_:y_test})
    print("epoch:",epoch+1,"accuracy",acc)

    
acc = sess.run(accuracy,feed_dict={x:X_test, y_:y_test})
print("final accuracy:",acc)



# evaluate test dataset

test = pd.read_csv('../input/test.csv')

test_data = test.values

predictions = sess.run(tf.argmax(ypred,1),feed_dict={x: test_data})

submission = pd.DataFrame(predictions,index=test.index+1)

submission.columns=["Label"]

submission.to_csv("my_sub.csv",index_label="imageId")
