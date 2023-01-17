# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

import tensorflow as tf



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
testC = test
train.head()
train.shape
target = train["label"]

train = train.drop("label",axis=1)
plt.figure(figsize=(10,10))

rows = 10

cols=10

for i in range(rows*cols):

    plt.subplot(rows,cols,i+1)

    plt.imshow(train.iloc[i].values.reshape(28,28),cmap="afmhot",interpolation="none")

    plt.xticks([])

    plt.yticks([])

plt.tight_layout()   
from sklearn.preprocessing import StandardScaler

train_std = StandardScaler().fit_transform(train)

test_std = StandardScaler().fit_transform(test)
train_mean = np.mean(train_std,axis=0)

train_cov = np.dot(((train_std - train_mean).T),(train_std-train_mean))

eign_value,eign_vector = np.linalg.eig(train_cov)

# eign_value_corr, eign_vector_corr = np.linalg.eig(train_corr)
eign_pair = [(np.abs(eign_value[i]),eign_vector[:,i]) for i in range(len(eign_value))]

eign_pair.sort(key=lambda x:x[0],reverse=True)
tot_eignV  = np.sum(eign_value)

eign_contr = [(sorted(eign_value,reverse=True)[i]/tot_eignV)*100 for i in range(len(eign_value))]
plt.figure(figsize=(20,8))

plt.subplot(1,2,1)

plt.plot(eign_contr)

plt.title("Importance of principal components using eigen value") #higher the eigen_contr , higher importance

plt.xlabel("principal components")

plt.ylabel("Variance contribution")



plt.subplot(1,2,2)

df = pd.DataFrame({"variance_ratio":eign_contr[:100],"PCs":[i for i in range(100)]})

sns.barplot(y= 'variance_ratio',x="PCs",data=df)

plt.xticks([])

plt.title("Scree plot")

plt.show()
print("First 50 Variables describe the variance proportion:%0.2f"%np.sum(eign_contr[:50])+"%")

print("First 200 Variables describe the variance proportion:%0.2f"%np.sum(eign_contr[:200])+"%")

print("First 400 Variables describe the variance proportion:%0.2f"%np.sum(eign_contr[:400])+"%")

plt.figure(figsize=(45,25))

for i in range(50):

    plt.subplot(5,10,i+1)

    plt.imshow((eign_pair[i][1]*eign_pair[i][0]).reshape(28,28),cmap="afmhot")

    plt.xticks([])

    plt.yticks([])

    plt.title("EigenValue"+str(i+1))

plt.show()
w  = np.hstack([eign_pair[i][1].reshape(784,1) for i in range(10) ])
X_projected = np.dot(train_std,w)
import plotly.graph_objs as go

import plotly.offline as py

py.init_notebook_mode(connected=True)

scatter_data = go.Scatter(

    x = X_projected[:6000][:,0],

    y = X_projected[:6000][:,1],

    mode = 'markers',

    text = target[:6000],

    showlegend = True,

    marker = dict(

        size = 8,

        color = target[:6000],

        colorscale ='Jet',

        showscale = False,

        line = dict(

            width = 2,

            color = 'rgb(255, 255, 255)'

        ),

        opacity = 0.8

    )

)

data = [scatter_data]



layout = go.Layout(

    title= 'PCA',

    hovermode= 'closest',

    xaxis= dict(

         title= 'PC1',

        ticklen= 5,

        zeroline= False,

        gridwidth= 2,

    ),

    yaxis=dict(

        title= 'PC2',

        ticklen= 5,

        gridwidth= 2,

    ),

    showlegend= False

)



fig = dict(data=data, layout=layout)

py.iplot(fig,filename="scatter-plot")
train = train/255.0

test = test/255.0
def createPlaceholders(n_h,n_w,n_c,n_y):

    X = tf.placeholder(np.float32,name="Xtrain",shape=[None,n_h,n_w,n_c])

    Y = tf.placeholder(np.float32,name="trueLabel",shape=[None,n_y])

    return X,Y
def initialize_parameters():



    w1 = tf.get_variable("w1", [3,3,1,8],initializer = tf.contrib.layers.xavier_initializer(seed = 0))

    w2 = tf.get_variable("w2", [3,3,8,16],initializer = tf.contrib.layers.xavier_initializer(seed = 0))

    parameters = {"W1": w1,

                  "W2": w2}

    return parameters
def forward_propagate(X,parameters):

    w1 = parameters["W1"]

    w2 = parameters["W2"]

    layer1 = tf.nn.conv2d(X,w1,strides=[1,1,1,1],padding="SAME")

    layer1_activation = tf.nn.relu(layer1)

    layer1_output = tf.nn.max_pool(layer1_activation,ksize=[1,4,4,1],strides=[1,4,4,1],padding="SAME")

    layer2 = tf.nn.conv2d(layer1_output,w2,strides=[1,1,1,1],padding="SAME")

    layer2_activation = tf.nn.relu(layer2)

    layer2_output = tf.nn.max_pool(layer2_activation,ksize=[1,4,4,1],strides=[1,4,4,1],padding="SAME")

    fltn = tf.contrib.layers.flatten(layer2_output)

    fully_con1 = tf.contrib.layers.fully_connected(fltn,100)

    z3 = tf.contrib.layers.fully_connected(fully_con1,10,activation_fn=None)

    return z3
def compute_cost(z3,y_true):

    softmax_loss = tf.nn.softmax_cross_entropy_with_logits(logits=z3,labels=y_true)

    cost  = tf.reduce_mean(softmax_loss)

    return cost
test = np.reshape(test.values,(-1,28,28,1))

test.shape
test = test.astype(np.float32)
from sklearn.model_selection import train_test_split

trainx,testx,trainy,testy = train_test_split(train,target,test_size=0.3)
# classifiers  = []

# from sklearn.linear_model import LogisticRegression

# from sklearn.tree import DecisionTreeClassifier

# from sklearn.ensemble import ExtraTreesClassifier

# from sklearn.ensemble import RandomForestClassifier

# from sklearn.ensemble import BaggingClassifier

# from sklearn.ensemble import AdaBoostClassifier

# from sklearn.ensemble import GradientBoostingClassifier
# seed = 0

# lr = LogisticRegression(random_state=seed,multi_class="multinomial",solver="lbfgs")

# dt = DecisionTreeClassifier()

# et  = ExtraTreesClassifier(verbose=1)

# rf = RandomForestClassifier(verbose=1)

# abc = AdaBoostClassifier()

# gb = GradientBoostingClassifier(verbose=1)

# classifiers.extend([lr,dt,et,rf,abc,gb])
# from sklearn.model_selection import KFold, cross_val_score,cross_validate

# def modelScore(model,X,y):

#     cv = KFold(5,random_state=seed,shuffle=True).get_n_splits()

#     score  = cross_validate(model,X,y,cv = cv,verbose=1)

#     return score
# trainScore = []

# valScore=[]

# std = []

# for model in classifiers:

#     score = modelScore(model,train,target)

#     trainScore.append(score["train_score"].mean())

#     valScore.append(score["test_score"].mean())

#     std.append(score["test_score"].std())

    
from keras.utils import to_categorical

trainx = np.reshape(trainx.values,[-1,28,28,1])

testx = np.reshape(testx.values,[-1,28,28,1])

trainy  = to_categorical(trainy.values,10)

testy = to_categorical(testy.values,10)

trainy.shape
def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):



    m = X.shape[0]                

    mini_batches = []

    np.random.seed(seed)

    permutation = list(np.random.permutation(m))

    shuffled_X = X[permutation,:,:,:]

    shuffled_Y = Y[permutation,:]



    num_complete_minibatches = math.floor(m/mini_batch_size)

    for k in range(0, num_complete_minibatches):

        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]

        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]

        mini_batch = (mini_batch_X, mini_batch_Y)

        mini_batches.append(mini_batch)

  

    if m % mini_batch_size != 0:

        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]

        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]

        mini_batch = (mini_batch_X, mini_batch_Y)

        mini_batches.append(mini_batch)

    

    return mini_batches

from tensorflow.python.framework import ops

import math

def model_evaluate(trainX,trainY,testX,testY,predictX,alpha= 0.01,mini_batch = 64,num_epochs=20,print_cost=True):

    ops.reset_default_graph()                       

    tf.set_random_seed(1)            

    seed = 3     

    (m, n_H0, n_W0, n_C0) = trainX.shape             

    n_y = trainY.shape[1]                            

    costs = []                

    X, Y = createPlaceholders(n_H0,n_W0,n_C0,n_y)

    parameters = initialize_parameters()

    Z3 = forward_propagate(X, parameters)

    cost = compute_cost(Z3,Y)

  

    optimizer = tf.train.AdamOptimizer(learning_rate=alpha).minimize(cost)

  

    init = tf.global_variables_initializer()

 

    with tf.Session() as sess:

        sess.run(init)

        for epoch in range(num_epochs):

            minibatch_cost = 0.

            num_minibatches = int(m / mini_batch) 

            seed = seed + 1

            minibatches = random_mini_batches(trainX, trainY, mini_batch, seed)

            for minibatch in minibatches:

                (minibatch_X, minibatch_Y) = minibatch

                _ , temp_cost = sess.run([optimizer,cost],feed_dict={X:minibatch_X,Y:minibatch_Y})

                minibatch_cost += temp_cost 

            minibatch_cost = minibatch_cost/num_minibatches

            if print_cost == True and epoch % 5 == 0:

                print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))

            if print_cost == True and epoch % 1 == 0:

                costs.append(minibatch_cost)

        plt.plot(np.squeeze(costs))

        plt.ylabel('cost')

        plt.xlabel('iterations (per tens)')

        plt.title("Learning rate =" + str(alpha))

        plt.show()

        

        predict_op = tf.argmax(Z3, 1)

        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print(accuracy)

        train_accuracy = accuracy.eval({X: trainX, Y: trainY})

        test_accuracy = accuracy.eval({X: testX, Y: testY})

        print("Train Accuracy:", train_accuracy)

        print("Test Accuracy:", test_accuracy)

        test_prediction = predict_op.eval(feed_dict={X:predictX})

        

        return train_accuracy, test_accuracy, parameters,test_prediction
_, _, parameters,predictions= model_evaluate(trainx,trainy,testx,testy,test,alpha= 0.001,mini_batch = 64,num_epochs=85,print_cost=True)
predictions[:10]
plt.figure(figsize=(10,10))

rows = 10

cols=10

for i in range(rows*cols):

    plt.subplot(rows,cols,i+1)

    plt.imshow(testC.iloc[i].values.reshape(28,28),cmap="afmhot",interpolation="none")

    plt.xticks([])

    plt.yticks([])

plt.tight_layout()   
submission = pd.DataFrame({"ImageId":[i+1 for i in range(len(predictions))],"Label":predictions})
submission.to_csv("submission.csv",index=False)