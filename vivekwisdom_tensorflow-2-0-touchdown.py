from IPython.display import IFrame

IFrame(width="560", height="315", src="https://www.youtube.com/embed/EqWsPO8DVXk", allowfullscreen="allowfullscreen", frameborder="0", allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture;")
# Optional we might have to force uninstall

# !pip uninstall -y tensorflow
import pip

pip.__version__
# Current stable release for CPU-only

!pip install tensorflow==2.0.0
import tensorflow as tf

print(tf.__version__)
from __future__ import absolute_import, division, print_function, unicode_literals



import tensorflow as tf



from tensorflow import keras
from tensorflow.keras import layers



model = tf.keras.Sequential([

# Adds a densely-connected layer with 64 units to the model:

layers.Dense(64, activation='relu', input_shape=(32,)),

# Add another:

layers.Dense(64, activation='relu'),

# Add a softmax layer with 10 output units:

layers.Dense(10, activation='softmax')])



model.summary()



model.compile(optimizer=tf.keras.optimizers.Adam(0.01),

              loss='categorical_crossentropy',

              metrics=['accuracy'])



from __future__ import absolute_import, division, print_function, unicode_literals



import tensorflow as tf



import cProfile
# Code to check if eager execution is enabbled by default



tf.executing_eagerly()
# Test Example Executes and returns results immediately



x = [[2., 2.], [2., 2.]]

y = [[4., 4.], [4., 4.]]

m = tf.matmul(x, y)

print(m)