# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
dataset = pd.read_csv("../input/creditcard.csv")

print(dataset.head())

print(dataset.describe())



print(len(dataset[dataset.Class == 1]))

features = dataset.iloc[:, :-1]

print(features.shape)

label = dataset.iloc[:, -1].values

print(label.shape)



# heatmap for correlation, verifying that pca is already done

import seaborn as sns

corrMat = features.corr()

sns.heatmap(corrMat, vmax=0.8)
from sklearn.preprocessing import StandardScaler



fraudInd = np.asarray(np.where(label == 1))

noFraudInd = np.where(label == 0)

features = features.values



# data standarization (zero-mean, unit variance) ~ truncation to [-1, 1]

scaler = StandardScaler()

scaler.fit(features)

features = scaler.transform(features)



#import matplotlib.pyplot as plt

#fig = plt.figure()

#ax1 = fig.add_subplot(221)

#ax1.hist(features[noFraudInd,0], 50)

#ax2 = fig.add_subplot(222)

#ax2.hist(features[noFraudInd,1], 50)

#ax3 = fig.add_subplot(223)

#ax3.hist(features[noFraudInd,2], 50)

#ax4 = fig.add_subplot(224)

#ax4.hist(features[noFraudInd,3], 50)
ind_fraud = np.where(label==1)

ind_noFraud = np.where(label==0)



features_noFraud = features[ind_noFraud]

features_fraud = features[ind_fraud]

print(features_noFraud.shape, features_fraud.shape, features.shape)



def get_mini_batch(x,y):

	rows=np.random.choice(x.shape[0], 100)

	return x[rows], y[rows]



import tensorflow as tf



label_noFraud = np.matmul(np.ones((features_noFraud.shape[0], 1)), np.array([0, 1]).reshape((1,2)))

label_fraud = np.matmul(np.ones((features_fraud.shape[0], 1)), np.array([1, 0]).reshape((1,2)))
from sklearn.model_selection import train_test_split

TestPortion = 0.2

RND_STATE = 1



# It is important to take same test_size and random_state for comparison

x_trainNF, x_testNF, y_trainNF, y_testNF = train_test_split(features_noFraud, label_noFraud, test_size=TestPortion, random_state=RND_STATE)

x_trainF, x_testF, y_trainF, y_testF = train_test_split(features_fraud, label_fraud, test_size=TestPortion, random_state=RND_STATE)



print(y_trainF.shape, y_trainNF.shape)

#y = np.append(y_trainNF, y_trainF)



# Now stack them together and permute

x_train = np.random.permutation(np.vstack((x_trainNF, x_trainF)))

y_train = np.random.permutation(np.vstack((y_trainNF, y_trainF)))

#y_train = np.random.permutation(np.append(y_trainNF, y_trainF))

x_test = np.random.permutation(np.vstack((x_testNF, x_testF)))

y_test = np.random.permutation(np.vstack((y_testNF, y_testF)))

#y_test = np.random.permutation(np.append(y_testNF, y_testF))

print(y_train.shape, y_test.shape)

print(x_train.shape, x_test.shape)
nFeature = x_train.shape[1]



# place holder for inputs. 

x = tf.placeholder("float", [None, nFeature])



# take 'nFeautre' features to 'N2' nodes in hidden layer,  

# init weights with truncated normal distribution

N2 = 100

w1 = tf.Variable(tf.truncated_normal([nFeature, N2], stddev=0.01, name = 'w1')) 

b1 = tf.Variable(tf.constant(0.0, shape=[N2])) 



# calculate activations 

hidden_output = tf.nn.relu(tf.matmul(x, w1) + b1)

#hidden_output = x



# bring from 'N2' nodes to 2 outputs:

#N3 = 10

w2 = tf.Variable(tf.truncated_normal([N2, 2], stddev=0.01, name = 'w2')) 

b2 = tf.Variable(tf.constant(0.0, shape=[2])) 



#hidden_output = tf.nn.relu(tf.matmul(hidden_output1, w2) + b2)

#w3 = tf.Variable(tf.truncated_normal([N3, 2], stddev=0.025, name = 'w3')) 

#b3 = tf.Variable(tf.constant(0.05, shape=[2])) 



# placeholder for labels

y_ = tf.placeholder("float", [None,2])



# #implement model. these are predicted ys

#y = tf.nn.sigmoid(tf.matmul(x, w1) + b1)

y = tf.nn.softmax(tf.matmul(hidden_output, w2) + b2)

loss = tf.reduce_mean(tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(y, y_, name='xentropy')))
opt = tf.train.AdamOptimizer(learning_rate=1e-4)

#opt = tf.train.GradientDescentOptimizer(learning_rate=0.001)

train_step = opt.minimize(loss, var_list=[w1,b1,w2,b2])

tf_correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

tf_accuracy = tf.reduce_mean(tf.cast(tf_correct_prediction, "float"))
# Start an interactive session

sess = tf.InteractiveSession()



# init all vars

init = tf.global_variables_initializer()

sess.run(init)
ntrials = 5000

for i in range(ntrials):

    # get mini batch

    a,b=get_mini_batch(x_train,y_train)

    

    # run train step, feeding arrays of 100 rows each time

    _, cost =sess.run([train_step,loss], feed_dict={x: a, y_: b})

    if i%100 ==0:

        trainAccuracy = tf_accuracy.eval(feed_dict={x: a, y_: b})  

        print("epoch is {0} and cost is {1} with train accuracy {2}".format(i, cost, trainAccuracy))   
result = tf_accuracy.eval(feed_dict={x: x_test, y_: y_test})



print("Test accuracy: {}".format(result))



# predicted labels

prob=y

y_pred = prob.eval(feed_dict={x:x_test, y_:y_test})

y_pred = np.argmax(y_pred, 1)

y_test = np.argmax(y_test, 1)



from sklearn.metrics import precision_recall_curve, auc, confusion_matrix, classification_report, recall_score, roc_auc_score

precision, recall, thresholds = precision_recall_curve(y_test, y_pred)

area = auc(recall, precision)

print('cm:', confusion_matrix(y_test,y_pred))

print('cr:', classification_report(y_test,y_pred))

print('recall_score:', recall_score(y_test,y_pred))

print('roc_auc_score:',roc_auc_score(y_test,y_pred))



sess.close()
x_tr, x_test, y_tr, y_test = train_test_split(features, label, test_size = TestPortion, random_state = 1)



from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(C = .01, penalty = 'l1', class_weight='balanced')

logreg.fit(x_tr,y_tr)

y_pred= logreg.predict(x_test)

print('------------ Results for LogiRegression ---------------')

print('cm:', confusion_matrix(y_test,y_pred))

#print('cr:', classification_report(y_test,y_pred))

#print('recall_score:', recall_score(y_test,y_pred))

print('roc_auc_score:',roc_auc_score(y_test,y_pred))



precision, recall, thresholds = precision_recall_curve(y_test, y_pred)

area = auc(recall, precision)

print("Area Under P-R Curve: ",area)



from sklearn.ensemble import ExtraTreesClassifier

clf = ExtraTreesClassifier(n_estimators =100)

clf.fit(x_tr, y_tr)

y_pred = clf.predict(x_test)

precision, recall, thresholds = precision_recall_curve(y_test, y_pred)

area = auc(recall, precision)



print('------------ Results for ExtraTreeClassifier ---------------')

print('cm:', confusion_matrix(y_test,y_pred))

#print('cr:', classification_report(y_test,y_pred))

#print('recall_score:', recall_score(y_test,y_pred))

print('roc_auc_score:',roc_auc_score(y_test,y_pred))

print("Area Under P-R Curve: ",area)



from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100)    

clf.fit(x_tr, y_tr)

importances = clf.feature_importances_

#plt.figure()

#plt.title("Feature importances")

#plt.bar(range(x_tr.shape[1]), importances,

#       color="r", align="center")

#plt.xticks(range(x_tr.shape[1]))

#plt.xlim([-1, x_tr.shape[1]])

#plt.show()



y_pred = clf.predict(x_test)

precision, recall, thresholds = precision_recall_curve(y_test, y_pred)

area = auc(recall, precision)



print('------------ Results for RandomForestClassifier ---------------')

print('cm:', confusion_matrix(y_test,y_pred))

#print('cr:', classification_report(y_test,y_pred))

#print('recall_score:', recall_score(y_test,y_pred))

print('roc_auc_score:',roc_auc_score(y_test,y_pred))

print("Area Under P-R Curve: ",area)
