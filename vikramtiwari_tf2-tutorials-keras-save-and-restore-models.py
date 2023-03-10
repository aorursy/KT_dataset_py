from __future__ import absolute_import, division, print_function

import os
import sys
print("Python version:", sys.version)

import tensorflow as tf
print("TensorFlow version:", tf.__version__)
keras = tf.keras
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0
# Returns a short sequential model
def create_model():
    model = tf.keras.models.Sequential([
        keras.layers.Dense(512, activation=tf.keras.activations.relu, input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation=tf.keras.activations.softmax)
    ])
    
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])
    
    return model

# Create a basic model instance
model = create_model()
model.summary()
checkpoint_path = 'training_1/cp.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)

# create checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                save_weights_only=True,
                                                verbose=1)

model = create_model()
model.fit(train_images, train_labels, epochs=10,
         validation_data = (test_images, test_labels),
         callbacks = [cp_callback]) # pass callback to training
!ls {checkpoint_dir}
model = create_model()

loss, acc = model.evaluate(test_images, test_labels)
print('Untrained model, accuracy: {:5.2f}%'.format(100*acc))
model.load_weights(checkpoint_path)
loss, acc = model.evaluate(test_images, test_labels)
print('Restored model, accuracy: {:5.2f}%'.format(100*acc))
# include the epoch in the file name. (use `str.format`)
checkpoint_path = 'training_2/cp-{epoch:04d}.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback =  tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 verbose=1,
                                                 save_weights_only=True,
                                                 period=5) #  save weights everry 5 epochs

model = create_model()
model.save_weights(checkpoint_path.format(epoch=0))
model.fit(train_images, train_labels,
         epochs=50, callbacks=[cp_callback],
         validation_data =  (test_images, test_labels),
         verbose=0)
!ls {checkpoint_dir}
latest = tf.train.latest_checkpoint(checkpoint_dir)
latest
# save the weights
model.save_weights('./checkpoints/my_checkpoint')

# restore the weights
model = create_model()
model.load_weights('./checkpoints/my_checkpoint')

loss, acc = model.evaluate(test_images, test_labels)
print('Restored model, accuracy: {:5.2f}%'.format(100*acc))
model = create_model()

# You need to use a keras.optimizer to restore the optimizer state from and HDF5 file.
model.compile(optimizer='adam',
             loss=tf.keras.losses.sparse_categorical_crossentropy,
             metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)

# save entire model to a HDF5 file
model.save('my_model.h5')
# Recreate the exact same model, including weights and optimizer.
new_model = keras.models.load_model('my_model.h5')
new_model.summary()
loss, acc = new_model.evaluate(test_images, test_labels)
print('Restored model, accuracy: {:5.2f}%'.format(100*acc))