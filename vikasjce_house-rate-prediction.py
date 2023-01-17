import numpy as np # Required for All Math functions

import pandas as pd # Required for data Handling

import matplotlib.pyplot as plt # Required for Graphical representation of Data

%matplotlib inline

import tensorflow as tf # Import tensor flow
a = tf.constant(1)

b = tf.constant(2)

c = tf.add(a,b)

with tf.Session() as sess:

    result = sess.run(c)

    print(result)
x_data=np.linspace(0,10,1000000)

noise =np.random.randn(len(x_data))
b=5
y_true=(0.5)*(x_data)+b+noise

print(y_true)
x_df =pd.DataFrame(x_data,columns=['x_data'])

x_df.head()
y_df = pd.DataFrame(y_true,columns=['y_true'])

y_df.head()
my_data=pd.concat([x_df,y_df],axis=1)

my_data.head()
my_data.sample(n=250).plot(kind="scatter",x='x_data',y='y_true')
batch_size = 8

np.random.randn(2)
m = tf.Variable(initial_value=0.78305046)

b = tf.Variable(initial_value=-0.72926149)
xph = tf.placeholder(tf.float32)

yph = tf.placeholder(tf.float32)
y_model = m*xph+b
error = tf.reduce_sum(tf.square(y_model-yph))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(error)
init = tf.global_variables_initializer()

with tf.Session() as sess:

    sess.run(init)

    batches =1000

    for i in range(batches):

        rand_ind = np.random.randint(len(x_data),size = batch_size)

        feed = {xph:x_data[rand_ind],yph:y_true[rand_ind]}

        error_val,_,m_model,=sess.run([error,optimizer,m],feed_dict=feed)

        print(error_val)

    m_final,b_final = sess.run([m,b])

    print("m_final",m_final,"\nb_final",b_final)
y_hat = m_final*x_data+b_final
my_data.sample(250).plot(kind="scatter",x='x_data',y='y_true')

plt.plot(x_data,y_hat,'r')
feat_cols = [tf.feature_column.numeric_column('x',shape=(1))]
estimator = tf.estimator.LinearRegressor(feature_columns=feat_cols)
from sklearn.model_selection import train_test_split
x_train,y_train,y_train,y_test = train_test_split(x_data,y_true,test_size=0.3)
x_train.shape,y_train.shape,x_test.shape,y_test.shape
dataset = pd.read_csv("../input/train.csv")

print(dataset)
X=dataset.iloc[:,1:80].values

print(X)
y=dataset.iloc[:,80].values

print(y)