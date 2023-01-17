import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
x_data = pd.read_csv('../input/needed-data-of-pokemon/pokemon_feature.csv')
y_data = pd.read_csv('../input/needed-data-of-pokemon/pokemon_label.csv')
x_train,x_test,y_train,y_test = train_test_split(x_data,y_data,shuffle=True,test_size=0.25)
x=tf.placeholder('float',shape=([None,6]))
y=tf.placeholder('float',shape=([None,1]))
fp=tf.Variable(x_test)
fp=tf.cast(fp,'float')
w0 = tf.Variable(tf.random_normal([6,1]))
b0 = tf.Variable(tf.random_normal([1]))
out = tf.add(tf.matmul(x,w0),b0)
cost = tf.square(tf.subtract(out,y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
train = optimizer.minimize(cost)
with tf.Session() as sess:
    #initializing global vars
    sess.run(tf.global_variables_initializer())
    #setting epochs
    epochs = 50000
    #running for epoch times
    for i in range(epochs):
        #training one epoch
        sess.run(train,feed_dict={x:x_train,y:y_train})
        #printing errors to ... uhh... see
        if i % 2500 ==0:
            print('Error at ',i,':',np.mean(sess.run(cost,feed_dict={x:x_train,y:y_train})))
    
    #getting test data and doing magic to get output
    y_pred=sess.run(tf.add(tf.matmul(fp,w0),b0))
    vr=0
    y_test = np.array(y_test)
    min_y=min(y_pred)
    max_y=max(y_pred)
    #loop that makes output 1 and 0
    while vr < len(y_pred):
        y_pred[vr] = y_pred[vr] - min(y_pred)
        y_pred[vr] = y_pred[vr] / max(y_pred)
        if y_test[vr] < 0:
            y_test[vr] = 0
        if y_pred[vr] > 0.8:
            y_pred[vr] = 1
        else:
            y_pred[vr] = 0
        vr+=1
    vr=0
    count=0
    
    #loop to count accuracy on test data
    while vr < len(y_pred):
        if y_pred[vr] == y_test[vr]:
            count+=1
        vr+=1 
    print('Accuracy :',count/len(y_pred))
    
    fw,fb = sess.run([w0,b0])
    
predict_this_x = np.array([[35,55,40,50,50,90]],dtype='float32')
predicted_y = np.add(np.matmul(predict_this_x,fw),fb)
predicted_y = predicted_y - min_y
predicted_y = predicted_y / max_y
if predicted_y > 0.8:
    print('Seems Legendary !')
else:
    print('Nah, just need some more :/')