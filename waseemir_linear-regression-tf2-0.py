import numpy as np 

import tensorflow as tf

import matplotlib.pyplot as plt
observations=10000

xs=np.random.uniform(-10,20,observations)

zs=np.random.uniform(-20,25,observations)

inputs=np.column_stack((xs,zs))

noise=np.random.uniform(-1,1,observations)

targets=2*xs-3*zs+4+noise

targets=targets.reshape(10000,1).round(1)

np.savez('data_set',input=inputs,target=targets)
data_set=np.load('data_set.npz')

data_set['target'].shape
input_size=2

output_size=1



model=tf.keras.Sequential([

    tf.keras.layers.Dense(output_size,

                         kernel_initializer=tf.random_uniform_initializer(-0.1,0.1),

                         bias_initializer=tf.random_uniform_initializer(-0.1,0.1),

                         )

])



custom_optimizer=tf.keras.optimizers.SGD(learning_rate=0.002)

model.compile(custom_optimizer,'mean_squared_error')

model.fit(data_set['input'],data_set['target'],epochs=10,verbose=2)
weights=model.layers[0].get_weights()[0]

biases=model.layers[0].get_weights()[1]

weights
biases
predict=np.array(model.predict_on_batch(data_set['input']))

predict.round(1)
data_set['target']
plt.plot(np.squeeze(model.predict_on_batch(data_set['input'])),np.squeeze(data_set['target']))

plt.xlabel('Inputs')

plt.ylabel('Targets')

plt.show()