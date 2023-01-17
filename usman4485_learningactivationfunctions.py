import tensorflow as tf

import numpy as np
W = tf.Variable(np.random.randn(2,1),dtype=np.float16)

b = tf.Variable([0.0],shape=1,dtype = np.float16)
W,b
@tf.function

def forward(X):

    output = tf.matmul(X,W)+b

    return output 
X = tf.convert_to_tensor(np.random.randn(1,2),dtype=np.float16)

print(X.numpy())
print(forward(X).numpy())
out = forward(X)

print(out)
sigmoid = tf.nn.sigmoid(out)

print(sigmoid.numpy())
tanh = tf.nn.tanh(out)
print(tanh.numpy())
relu = tf.nn.relu(out)

print(relu.numpy())
softmax = tf.nn.softmax(out)

print(softmax.numpy())