
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import tensorflow as tf
import tarfile 
from subprocess import check_output
import matplotlib.pyplot as plt
def convert_csv_numpy_array(filepath, delimiter):
    return np.genfromtxt(filepath,delimiter=delimiter) #dtype is none

def exract_data():
#     if "../input/data" not in os.listdir(os.getcwd()):
        
#         print('ppppppp',check_output(["ls", "../input"]))
#         tarSet= tarfile.open("../input/data.tar.gz","r:gz")
#         tarSet.extractall()
#         tarSet.close()
        
#     else:
    print("Already data file is there")
    
    x_train= convert_csv_numpy_array("../input/data/data/trainX.csv", delimiter="\t")
    y_train= convert_csv_numpy_array("../input/data/data/trainY.csv", delimiter="\t")
    x_test= convert_csv_numpy_array("../input/data/data/testX.csv", delimiter="\t")
    y_test= convert_csv_numpy_array("../input/data/data/testY.csv", delimiter="\t")
    return x_train,y_train,x_test,y_test
x_train,y_train,x_test,y_test = exract_data()
# print("fnsss  ",str(os.listdir("../input/data/data")) )
# for fn in os.listdir("../input/data"):
#     print("fnsss  "+fn)
# if tarfile.is_tarfile("../input/data/data"):
#     print("prrrrrrr")
# else:
#     print("Notttttttttt")
x_arr= np.asarray(x_train, dtype='float32')
print(x_arr.shape[0])
input_features= x_train.shape[1]
output_lables= y_train.shape[1]

print(x_train.shape[0],"shape", input_features)
print( y_train.shape[0],"shape", output_lables)

x= tf.placeholder(tf.float32, [None, input_features])

y= tf.placeholder(tf.float32, [None, output_lables])

weight= tf.Variable(tf.random_normal([input_features,output_lables],
                                       mean=0,
                                       stddev=(np.sqrt(6/(input_features+
                                                         output_lables+1))
                                       )), name="weight")

bias= tf.Variable(tf.random_normal([1,output_lables],
                                       mean=0,
                                       stddev=(np.sqrt(6/(input_features+
                                                         output_lables+1))
                                       )),name="bias")
#cal

#apply weights and bias
#sigmoid is activation function: affine and activation for each cell
layer1= tf.nn.sigmoid(tf.add(tf.matmul(x, weight),bias ), name="layer")

cost= tf.losses.mean_squared_error(labels=y,predictions=layer1)

optimizer= tf.train.GradientDescentOptimizer(0.001).minimize(cost)


#plot graph and show there

epoch_values=[]
accuracy_values=[]
cost_values=[]

# Turn on interactive plotting
plt.ion()

# Create the main, super plot
fig = plt.figure()

ax1 = plt.subplot("211")
ax1.set_title("TRAINING ACCURACY", fontsize=18)
ax2 = plt.subplot("212")
ax2.set_title("TRAINING COST", fontsize=18)
plt.tight_layout()

sess = tf.Session()
sess.run( tf.initialize_all_variables())

correct_prediction_op= tf.equal(tf.argmax(layer1,1), tf.argmax(y,1))

acuuracy_op= tf.reduce_mean(tf.cast(correct_prediction_op, "float"))


activation_summary_OP = tf.summary.histogram("output", layer1)


accuracy_summary_OP = tf.summary.scalar("accuracy", acuuracy_op)
cost_summary_OP = tf.summary.scalar("cost", cost)


weightSummary = tf.summary.histogram("weights", weight.eval(session=sess))
biasSummary = tf.summary.histogram("biases", bias.eval(session=sess))

all_summary_OPS = tf.summary.merge_all()

writer = tf.summary.FileWriter("summary_logs", sess.graph)

costDiff=1
initCost=0
for i in range(1000):
    
    if i>1 and costDiff < 0.001 :
        print("Stop training")
        break;
    else:
        
         optValue = sess.run(optimizer, feed_dict={x: x_train, y: y_train})
        
#          if i % 10 == 0:
            
#             summ_res, accur, currCost= sess.run([all_summary_OPS, acuuracy_op, cost],feed_dict={x: x_train, y: y_train})
            
#             writer.add_summary(summ_res, i)
#             accuracy_values.append(accur)
#             cost_values.append(currCost)
            
#             costDiff = abs(currCost - initCost)
#             initCost=currCost
            
#             print("step %d, training accuracy %g"%(i, accur))
#             print("step %d, cost %g"%(i, currCost))
#             print("step %d, change in cost %g"%(i, costDiff))
            
#             #plot
            
#             ax1.plot(epoch_values, accuracy_values)
            
#             ax2.plot(epoch_values, cost_values)
            
#             fig.canvas.draw()
            
            
print("final accuracy on test set: %s" %str(sess.run(acuuracy_op, 
                                                     feed_dict={x: x_test, 
                                                                y: y_test})))


model_saver= tf.train.Saver()

model_saver.save(sess,"../input/trained_model.ckpt")

sess.close()