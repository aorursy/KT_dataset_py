import tensorflow as tf

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.metrics import confusion_matrix

import warnings

warnings.simplefilter('ignore')
from tensorflow.examples.tutorials.mnist import input_data

data = input_data.read_data_sets("data/MNIST/",one_hot=True)
train = data.train.images

train_labels = data.train.labels

test = data.test.images

test_labels = data.test.labels
print("shape of train ",train.shape)

print("shape of test ",test.shape)
def plot_images(images,true_label,pred_label=None,missclassification=False):

  length = np.shape(images)[0]

  assert length==9

  fig = plt.figure(figsize=(7,7))

  ax = fig.subplots(3,3)

  for i,a in enumerate(ax.flat):

    a.imshow(np.reshape(images[i],(28,28)))

    if(pred_label==None):

      xlabel = "true label:{}".format(true_label[i])

    if(pred_label!=None):

      xlabel = "true label:{}, pred label:{}".format(true_label[i],pred_label[i])

    a.set_xlabel(xlabel)

    a.set_xticks([])

    a.set_yticks([])

    a.set_xlabel(xlabel)

  if(missclassification==True):

    fig.suptitle('Missclassified Examples',fontsize=20)

  else:

    fig.suptitle("Examples from dataset",fontsize=20)

  plt.show()

 
plot_images(train[0:9,:],list(np.argmax(train_labels[0:9,:],axis=1)))
def initialize_placeholders(X,Y):

  x = tf.placeholder(tf.float32,[None,X.shape[1]])

  y = tf.placeholder(tf.float32,[None,Y.shape[1]])

  return x,y
x,y = initialize_placeholders(train,train_labels)
def weight_initialize(X,Y):

  parameters ={}

  input_flat_shape = X.get_shape().as_list()[1]

  class_shape = Y.get_shape().as_list()[1]

  W1 = tf.get_variable("W1",initializer=tf.zeros([input_flat_shape,class_shape]))

  b1 = tf.get_variable("B1",initializer=tf.ones([class_shape]))

  parameters['w1'] = W1

  parameters['b1'] = b1

  return parameters

  

  
parameters = weight_initialize(x,y)
def logits(parameters,X,y):

  W1 = parameters['w1']

  b1 = parameters['b1']

  logit_ =tf.matmul(X,W1)+b1

  return logit_
logit = logits(parameters,x,y)

y_pred = tf.nn.softmax(logit)
logit
def loss(logit,y):

  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = logit,labels =y)

  cost = tf.reduce_mean(cross_entropy)

  return cost
loss = loss(logit,y)
def optimize(cost,learning_rate=0.01):

  optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

  return optimizer
optimizer = optimize(loss,0.01)
def Accuracy(y_pred,y):

  acc = tf.equal(tf.argmax(y_pred,dimension=1),tf.argmax(y,dimension=1))

  accuracy = tf.reduce_mean(tf.cast(acc,tf.float32))

  return accuracy


accuracy = Accuracy(y_pred,y)
import math

def batches_tuples(X,y,mini_batch=64):

  batches =[]

  num_batches = math.floor(X.shape[0]/mini_batch)

  for i in range(num_batches-1):

    (x_batch,y_batch) = X[i*mini_batch:(i+1)*mini_batch,:],y[i*mini_batch:(i+1)*mini_batch,:]

    batch = (x_batch,y_batch)

    batches.append(batch)

    

  (x_batch,y_batch)= X[(i+1)*mini_batch:X.shape[0],:],y[(i+1)*mini_batch:y.shape[0],:]

  batches.append((x_batch,y_batch))

  return batches

  

    
sess = tf.Session()
def model_run(mini_batch=64,num_epochs=1):

  init = tf.global_variables_initializer()

  batches = batches_tuples(train,train_labels,mini_batch=64)

  saver = tf.train.Saver()

  sess.run(init)

  cost_history = []

  acc_history = []

  test_acc_history =[]

  for i in range(num_epochs):

      total_cost = 0

      for j in range(len(batches)):

        (x_mini_train,y_mini_train) = batches[j]

        #print(x_mini_train.shape,y_mini_train.shape)

        _,c = sess.run([optimizer,loss],feed_dict = {x:x_mini_train,y:y_mini_train})

        total_cost = total_cost+c

      avg_cost = total_cost/len(batches)

      sess.run(y_pred,feed_dict={x:train,y:train_labels})

      acc = sess.run(accuracy,feed_dict={x:train,y:train_labels})

      test_acc = sess.run(accuracy,feed_dict={x:test,y:test_labels})

      cost_history.append(avg_cost)

      acc_history.append(acc)

      test_acc_history.append(test_acc)

      saver_path = saver.save(sess,"tmp/model_mnist.ckpt")

      if(i%5==0):

        print("epoch number:{}".format(i), "cost:{}".format(avg_cost),"train accuracy:{}".format(acc),"test_acc:{}".format(test_acc))

  print("epoch number:{}".format(i), "cost:{}".format(avg_cost),"train accuracy:{}".format(acc),"test_acc:{}".format(test_acc))      

  plt.plot(range(num_epochs),cost_history,"r-")

#   plt.plot(acc_history,"b-")

#   plt.plot(test_acc_history,"g-")

  plt.xlabel("Number of epochs")

  plt.ylabel("cost")

  plt.title("cost v/s epochs")

  plt.show()

  print("model saved in",saver_path)

  

        

        

      
model_run(64,25)
def plot_weights(img_shape,variable_name,model_path):

  trainable = tf.trainable_variables()

  w = [v for v in trainable if v.name==variable_name ]

  saver = tf.train.Saver()

  graph = tf.get_default_graph()

  with tf.Session(graph = graph) as sess:

    saver.restore(sess,model_path)

    print("model resotred")

    weights = sess.run(w)

  wmin = np.min(weights[0])

  wmax = np.max(weights[0])

  

  fig = plt.figure(figsize=(15,5))

  fig.subplots_adjust(hspace=0.3,wspace=0.3)

  axes = fig.subplots(3,4)

  for i,ax in enumerate(axes.flat):

    if(i>=10):

      ax.set_visible(False)

      continue

    

    ax.imshow(weights[0][:,i].reshape(img_shape),vmin=wmin,vmax=wmax,cmap='seismic')

    ax.set_xlabel("class:{}".format(i))

    ax.set_xticks([])

    ax.set_yticks([])

  plt.show()

plot_weights((28,28),"W1:0",'tmp/model_mnist.ckpt')


def print_confusion_matrix(X,Y):

  pred = sess.run(y_pred,feed_dict={x:X,y:Y})

  pred = np.argmax(pred,axis=1)

  cm = confusion_matrix(pred,np.argmax(Y,axis=1))

  plt.figure(figsize=(8,8))

  sns.heatmap(cm,annot=True,cmap='YlGnBu',fmt='d')

  plt.xlabel("predicted label")

  plt.ylabel("true label")

  plt.title("confusion matrix")

  plt.show()

  

    
print_confusion_matrix(train,train_labels)
print_confusion_matrix(test,test_labels)
def misclassified_example(X,Y):

  pred = sess.run(y_pred,feed_dict={x:X,y:Y})

  predictions = np.argmax(pred,axis=1)

  labels = np.argmax(Y,axis=1)

  #print(labels)

  bool_ar = sess.run(tf.equal(predictions,labels))

  bool_ar = bool_ar==False

  #print(bool_ar)

  bool_ar = list(bool_ar)

  index_arr  =[i for i in range(len(bool_ar)) if bool_ar[i]==True]

  #print(index_arr)

  example_missclassified = X[index_arr,:][0:9]

  #print(example_missclassified)

  true_labels = np.argmax(Y[index_arr,:],axis=1)

  #print(true_labels)

  true_labels=true_labels[0:9]

  #print(true_labels)

  predicted_labels = [predictions[i] for i in index_arr]

  predicted_labels =predicted_labels[:9]

  

  plot_images(example_missclassified,true_labels,predicted_labels,True)

  

  
misclassified_example(train,train_labels)
misclassified_example(test,test_labels)
train_feed_dict={x:train,y:train_labels}

test_feed_dict = {x:test, y:test_labels}
from sklearn.metrics import auc,roc_curve,classification_report
prob = y_pred.eval(session=sess,feed_dict =test_feed_dict)

tpr_per_class={}

fpr_per_class={}

auc_per_class={}

auc_per_class={}

thresholds={}

for i in range(10):

  fpr_per_class[i],tpr_per_class[i],thresholds[i] = roc_curve(test_labels[:,i],prob[:,i])

  auc_per_class[i] = auc(fpr_per_class[i],tpr_per_class[i])
def plot_roc(fpr,tpr):

  fig = plt.figure(figsize=(15,15))

  axes = fig.subplots(4,3)

  for i,ax in enumerate(axes.flat):

    if(i>=10):

      ax.set_visible(False)

      continue

    ax.plot(fpr[i],tpr[i],color="darkorange")

    ax.set_xlabel("FPR(false positive rate)")

    ax.set_ylabel("TPR(True positive rate)")

    ax.legend(["class:{}".format(i)+" ROC_curve(area:{0:.2f})".format(auc_per_class[i])],loc='lower left')

  fig.suptitle("ROC CURVES",fontsize=20)

  plt.show()

  

  print(classification_report(np.argmax(test_labels,axis=1),np.argmax(prob,axis=1)))
plot_roc(fpr_per_class,tpr_per_class)
sess.close()