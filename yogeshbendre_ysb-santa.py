# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import tensorflow as tf
from subprocess import check_output
import pickle as pkl
print(check_output(["ls", "../input"]).decode("utf8"))
print(check_output(["ls", "./"]).decode("utf8"))
#test_data=pd.read_csv('../input/test.csv')
test_data=np.genfromtxt('../input/test.csv',delimiter=',')
#train_data=pd.read_csv('../input/test.csv')
train_data=np.genfromtxt('../input/test.csv',delimiter=',')
#print(train_data)
pkl.dump(train_data[1:],open('train_data.pkl','wb'))
pkl.dump(test_data[1:],open('test_data.pkl','wb'))

trdt=pkl.load(open('train_data.pkl','rb'))
print('train_data:',len(trdt[1,:]))
trlb=trdt[:,-1]
trdt=trdt[:,:-1]
ntr=len(trdt)
print('train_data:',len(trdt[1,:]))
tsdt=pkl.load(open('test_data.pkl','rb'))
nts=len(tsdt)
#print(trdt[:,-1])
# Any results you write to the current directory are saved as output.

#Tensorflow code
graph=tf.Graph()
batch_size=128
with graph.as_default():
    tf_trdt=tf.placeholder(tf.float32,shape=(batch_size,ntr))
    tf_trlb=tf.placeholder(tf.float32,shape=(batch_size,2))
    tf_tsdt=tf.constant(tsdt)
    
    #Hidden Layer 1
    nh1=256
    w1=tf.Variable(tf.truncated_normal([ntr,nh1]))
    b1=tf.Variable(tf.zeros([nh1]))
    

    #Hidden Layer 2
    nh2=2 #Only two classes 
    w2=tf.Variable(tf.truncated_normal([nh1,nh2]))
    b2=tf.Variable(tf.zeros([nh2]))
    
    #Network Mapping
    h1=tf.nn.relu(tf.matmul(tf_trdt,w1)+b1)
    h2=tf.nn.relu(tf.matmul(h1,w2)+b2)
    
    logits=tf.matmul(h1,w2)+b2
    loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits,tf_trlb)) + 0.1*tf.nn.l2_loss(w1) + 0.01*tf.nn.l2_loss(w2)
    
    #Optimizer
    optimizer=tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    
    #Predictions
    trpr=tf.nn.softmax(logits)
    tspr=tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(tsdt,w1)+b1),w2)+b2)
    

#Tensorflow session to run    
num_steps=1001

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print("Initialized")
    for step in range(num_steps):
        offset=(step*batch_size)%(trlb.shape[0]-batch_size)
        batch_data=trdt[offset:(offset+batch_size),:]
        batch_labels=trlb[offset:(offset+batch_size),:]
        feed_dict={tf_trdt:batch_data,tf_trlb:batch_labels}
        _,l,pr=session.run([optimizer,loss,trpr],feed_dict=feed_dict)
        if(step%100==0):
            print('Minibatch loss at step %d : %f' % (step,l))
            print('Minibatch accuracy: %.lf%%' % accuracy(trpr,batch_labels))
            

    print('Minibatch accuracy: %.lf%%' % accuracy(tspr.eval(),tslb))

            
def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

        





