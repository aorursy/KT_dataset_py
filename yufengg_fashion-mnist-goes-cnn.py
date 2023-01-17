"""Convolutional Neural Network Custom Estimator for MNIST, built with tf.layers."""
import os
print(os.listdir("../input"))

import argparse
import os
import numpy as np
import time

import pandas as pd
import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data

BATCH_SIZE = 100
# To use 'fashion mnist', instead point DATA_DIR to a directory containing
# these files: https://github.com/zalandoresearch/fashion-mnist#get-the-data
# DATA_DIR = '/Users/yufengg/code/github/fashion-mnist-dataset'
MODEL_DIR = os.path.join("/tmp/tfmodels/fashion_mnist_cnn_estimator",
                          str(int(time.time())))

# This is too short for proper training (especially with 'Fashion-MNIST'), 
# but we'll use it here to make the notebook quicker to run.
NUM_STEPS = 1000

tf.logging.set_verbosity(tf.logging.INFO)
print("using model dir: %s" % MODEL_DIR)
print("Using TensorFlow version %s" % (tf.__version__)) 
data_train_file = "../input/fashion-mnist_train.csv"
data_test_file = "../input/fashion-mnist_test.csv"

df_train = pd.read_csv(data_train_file)
df_test = pd.read_csv(data_test_file)
def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""

  # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # MNIST images are 28x28 pixels, and have one color channel
  input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 28, 28, 1]
  # Output Tensor Shape: [batch_size, 28, 28, 32]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 28, 28, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 32]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 14, 14, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 64]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 14, 14, 64]
  # Output Tensor Shape: [batch_size, 7, 7, 64]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 7, 7, 64]
  # Output Tensor Shape: [batch_size, 7 * 7 * 64]
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 7 * 7 * 64]
  # Output Tensor Shape: [batch_size, 1024]
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu, name="dense1")

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 10]
  logits = tf.layers.dense(inputs=dropout, units=10)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  prediction_output = tf.estimator.export.PredictOutput({"classes": tf.argmax(input=logits, axis=1),
     "probabilities": tf.nn.softmax(logits, name="softmax_tensor")})

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions,
        export_outputs={tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: prediction_output})

  # Calculate Loss (for both TRAIN and EVAL modes)
  onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
  loss = tf.losses.softmax_cross_entropy(
      onehot_labels=onehot_labels, logits=logits)
  # Generate some summary info
  tf.summary.scalar('loss', loss)
  tf.summary.histogram('conv1', conv1)
  tf.summary.histogram('dense', dense)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
df_train[df_train.columns[1:]].values.shape

def get_input_fn(dataframe, batch_size=100, num_epochs=1, shuffle=True):
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": dataframe[dataframe.columns[1:]].values/255},
        y=dataframe["label"],
        batch_size=batch_size,
        num_epochs=num_epochs,
        shuffle=True)
    return input_fn

train_input_fn = get_input_fn(df_train, num_epochs=1)
eval_input_fn = get_input_fn(df_test, shuffle=False)
def generate_input_fn(dataset, batch_size=BATCH_SIZE):
    def _input_fn():
        X = tf.constant(dataset.images)
        Y = tf.constant(dataset.labels, dtype=tf.int32)
        image_batch, label_batch = tf.train.shuffle_batch([X,Y],
                               batch_size=batch_size,
                               capacity=8*batch_size,
                               min_after_dequeue=4*batch_size,
                               enqueue_many=True
                              )
        return {'x': image_batch} , label_batch

    return _input_fn
# Create the Estimator
mnist_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn, model_dir=MODEL_DIR)

# Set up logging for predictions
# Log the values in the "Softmax" tensor with label "probabilities"
tensors_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(
    tensors=tensors_to_log, every_n_iter=2000)
# Train the model
mnist_classifier.train(
    input_fn=train_input_fn
#     generate_input_fn(mnist.train, batch_size=BATCH_SIZE),
#     steps=NUM_STEPS
#     , hooks=[logging_hook]
    )
# Evaluate the model and print results
# eval_input_fn = tf.estimator.inputs.numpy_input_fn(
#     x={"x": eval_data},
#     y=eval_labels,
#     num_epochs=1,
#     shuffle=False)
eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
print(eval_results)
predict_input_fn = tf.estimator.inputs.numpy_input_fn(        
        x={'x': df_test[df_test.columns[1:]].iloc[5000:5005].values/255},
        batch_size=1,
        num_epochs=1,
        shuffle=False)


predictions = mnist_classifier.predict(input_fn=predict_input_fn)

for prediction in predictions:
    print("Predictions:    {} with probabilities {}\n".format(
        prediction["classes"], prediction["probabilities"]))
print('Expected answers values: \n{}'.format(
    df_test["label"].iloc[5000:5005]))
import matplotlib.pyplot as plt
%matplotlib inline

for i in range(5000,5005): 
    sample = np.reshape(df_test[df_test.columns[1:]].iloc[i].values/255, (28,28))
    plt.figure()
    plt.title("labeled class {}".format(df_test["label"].iloc[i]))
    plt.imshow(sample, 'gray')
