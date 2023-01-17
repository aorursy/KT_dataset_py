import os

from PIL import Image
import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm import tqdm, tqdm_notebook
import cv2

tf.logging.set_verbosity(tf.logging.INFO)
train_df = pd.read_csv('../input/train.csv')
train_df.head()
X_tr = []
Y_tr = []
imges = train_df['id'].values
train_dir = '../input/train/train/'
temp = cv2.imread('../input/train.zip/0004be2cfeaba1c0361d39e2b000257b.jpg')
open('../input/train/train/0004be2cfeaba1c0361d39e2b000257b.jpg')
#Image Augment 
for img_id in imges:
    image = np.array(cv2.imread(train_dir + img_id))
    
    
    label = train_df[train_df['id'] == img_id]['has_cactus'].values[0]

    X_tr.append(image)
    Y_tr.append(label)  

    X_tr.append(np.flip(image))
    Y_tr.append(label)  

    X_tr.append(np.flipud(image))
    Y_tr.append(label)  

    X_tr.append(np.fliplr(image))
    Y_tr.append(label)  
                
X_tr = np.asarray(X_tr).astype('float32')/225

Y_tr = np.asarray(Y_tr)
X_tr.shape,Y_tr.shape
from sklearn.model_selection import train_test_split

train_data, eval_data, train_labels, eval_labels = train_test_split(X_tr, Y_tr, test_size=0.3)
# Load training and eval data
'''
((train_data, train_labels), (eval_data, eval_labels)) = tf.keras.datasets.mnist.load_data()

train_data = train_data/np.float32(255)
train_labels = train_labels.astype(np.int32)  # not required

eval_data = eval_data/np.float32(255)
eval_labels = eval_labels.astype(np.int32)  # not required
'''
def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, 32, 32, 3])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense Layer
    n = pool2.shape
    pool2_flat = tf.reshape(pool2, [-1, 8 * 8 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=2)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        
        accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions["classes"], name='acc_op')
        metrics = {'accuracy': accuracy}
        tf.summary.scalar('accuracy', accuracy[1])
        
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"], name ='acc_val')
        }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
# Create the Estimator
cactus_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="./tmp/cactus_convnet_model")
# Set up logging for predictions
tensors_to_log = {"probabilities": "softmax_tensor"}

logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)
# Train the model
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": train_data},
    y=train_labels,
    batch_size=100,
    num_epochs=None,
    shuffle=True)

eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": eval_data},
    y=eval_labels,
    num_epochs=1,
    shuffle=False)
for i in range(10):
    cactus_classifier.train(input_fn=train_input_fn, steps=100)

    eval_results = cactus_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)
#!tensorboard --logdir=./tmp/cactus_convnet_model
from os import listdir
from os.path import isfile, join

mypath = '../input/test/test'
files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
pred = []

for img in files:
    image = np.array(cv2.imread(mypath + '/' + img))
    pred.append(image)

pred = np.asarray(pred).astype('float32')/225
pred.shape
pred_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": pred},
    num_epochs=1,
    shuffle=False)
result = cactus_classifier.predict(input_fn = pred_input_fn)
temp = [x['probabilities'][1] for x in result]
pd.DataFrame({'id': files, 'has_cactus':temp}).to_csv('submission.csv', index = False)