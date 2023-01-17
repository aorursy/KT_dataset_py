import numpy as np
import pandas as pd
import tensorflow as tf

# this is important line if you intend to monitor your training/evaluation metrics in tensorboard
# stuff like accuracy, loss and etc.
tf.logging.set_verbosity(tf.logging.INFO)

# helper method for loading and preprocessing the data
def load_datasets():
    '''
    Method for loading and preprocessing the data
    
    OUTPUT:
    np.array of training pixel data
    np.array of training labels
    np.array of inference pixel data
    '''
    # reading the Kaggle datasets
    train = pd.read_csv('../input/train.csv')
    infer = pd.read_csv('../input/test.csv')
    
    # print some summary for the datasets
    print('Reading Successful.')
    print('Size of the training dataset {}'.format(train.shape))
    print('Size of the inference dataset {}'.format(infer.shape))
    
    # helper function for normalization of the images and converting them to np.arrays
    def image_preprocessing(df):
        df = df / 255.0 # normalization
        df = df.values.tolist() # convert to list
        array = np.asarray(df, dtype=np.float32) # convert to numpy arrays
        
        return array
    
    # here we pass only the pixel data to our preprocessing method
    train_images = image_preprocessing(train.drop('label', axis=1))
    infer_images = image_preprocessing(infer)
    train_labels = np.asarray(train['label'],dtype=np.int32)
    
    print('Preprocessing Successful.')
    print('Dimensions of train_images array {}'.format(train_images.shape))
    print('Dimensions of infer_images array {}'.format(infer_images.shape))
    print('Dimensions of train_labels array {}'.format(train_labels.shape))
    
    return train_images, train_labels, infer_images

train_images, train_labels, infer_images = load_datasets()
'''
x - according to the documentation I am passing a dictionary of my training features
y - passing the training labels
batch_size - how many images I want to pass to my NN in a single training step. 
             The gradients and the loss for the training step will be calculated only on these images. 
num_epochs - when I want to complete the execution of this method. With the current setting (1) 
             when the input function goes through all images once it will complete.
shuffle    - do I want to read the images in order or no. It is a better strategy to 
             shuffle within the training images during training.
'''
train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={'x':train_images},
      y=train_labels,
      batch_size=80,
      num_epochs=1,
      shuffle=True)
'''
What the eval_input_fn will do with its current settings is to read the entire training dataset 
(in order) using batch_size of 128 (default setting). 

Keep in mind that it will also terminate when completes 1 epoch.
'''
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'x':train_images},
    y=train_labels,
    shuffle=False,
    num_epochs=1)
'''
What the predict_input_fn will do with its current settings is to read the entire inference dataset 
(in order) using batch_size of 128 (default setting). 

Keep in mind that it will also terminate when completes 1 epoch.
'''
predict_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'x':infer_images},
    shuffle=False,
    num_epochs=1)
def cnn_model_function(features, labels, mode):
    '''
    This is the function which describes the structure of the neural network
    '''
    
    # input layer
    # reshaping x to 4-D tensor: [batch_size, width, height, channels]
    # features['x'] - the dictionary we passed for x in the input functions
    layer_1 = tf.reshape(features['x'], [-1, 28, 28, 1]) 
    
    # convolution layer 1
    # computes 32 features using 10x10 filter with ReLU activation.
    # input tensor: [batch_size, 28, 28, 1]
    # output tensor: [batch_size, 28, 28, 32]
    layer_2 = tf.layers.conv2d(                          
                inputs=layer_1,
                filters=32,
                kernel_size=[10, 10],
                padding="same",
                activation=tf.nn.relu)
    
    # convolution layer 2
    # computes 32 features using 5x5 filter with ReLU activation.
    # input tensor: [batch_size, 28, 28, 32]
    # output tensor: [batch_size, 28, 28, 64]
    layer_3 = tf.layers.conv2d(
                inputs=layer_2,
                filters=64,
                kernel_size=[5, 5],
                padding="same",
                activation=tf.nn.relu)
    
    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, 28, 28, 64]
    # Output Tensor Shape: [batch_size, 28 * 28 * 64]
    layer_4 = tf.reshape(layer_3, [-1, 28 * 28 * 64])
    
    # Dense Layer
    # Densely connected layer with 1024 neurons
    # Input Tensor Shape: [batch_size, 28 * 28 * 64]
    # Output Tensor Shape: [batch_size, 1024]
    layer_5 = tf.layers.dense(inputs=layer_4, units=1024, activation=tf.nn.relu)
    
    # Dropout operation; 0.6 probability that element will be kept
    # notice that this layer will perform droupout only during training!
    layer_6 = tf.layers.dropout(inputs=layer_5, rate=0.4, training=(mode == tf.estimator.ModeKeys.TRAIN))
    
    # Logits layer
    # Input Tensor Shape: [batch_size, 1024]
    # Output Tensor Shape: [batch_size, 10]
    logits = tf.layers.dense(inputs=layer_6, units=10)
    
    # define the values which our neural network will output
    # classes - which number the NN 'thinks' is on the image
    # probabilities - how certain our NN is about its prediction
    predictions = {
                "classes": tf.argmax(input=logits, axis=1),
                "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    
    # here we define what happens if we call the predict method of our estimator
    # with the current settings it will return the dictionary defined above
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    # here we define the loss for our training (the thing we minimize)
    # I do not need to perform one-hot-encoding to my training labels because the method
    # sparse_softmax_cross_entropy will do that for me and I don't need to think about that
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    
    # here we define how we calculate our accuracy
    # if you want to monitor your training accuracy you need these two lines
    accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions['classes'], name='acc_op')
    tf.summary.scalar('accuracy', accuracy[1])
    
    # here we define what happens if we call the train method of our estimator
    # with its current settings it will adjust the weights and biases of our neurons
    # using the Adam Optimization Algorithm based on the loss function we defined earlier
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    
    # what evaluation metric we want to show
    eval_metric_ops = {"accuracy": accuracy}
    
    # here we define what happens if we call the evaluate method of our estimator
    # with its current settings it will display the loss and the accuracy which we defined earlier
    return tf.estimator.EstimatorSpec(
                mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
# first we define a folder where tensorflow will keep its progress
# this includes periodical saves of our weight, biases, accuracy, loss and etc.
# so if we have more training images we can simply continue training on them
OUTDIR = './CNN_CLASSIFIER'

# we create an estimator object which:
# - is using the Neural Net structure from the cnn_model_function
# - reads/writes the files written in the directory which we defined earlier
cnn_classifier = tf.estimator.Estimator(model_fn=cnn_model_function, model_dir=OUTDIR)

# here we start the FileWriter method which will actually save the progress in the folder defined above
file_writer = tf.summary.FileWriter(OUTDIR)

# a small helper function which trains/evaluates our network for a given number of epochs
# remember that our input functions go through the datasets only once
def train_and_evaluate(estimator, epochs=30):
    for i in range(epochs):
        estimator.train(input_fn=train_input_fn)
        estimator.evaluate(input_fn=eval_input_fn)

# and here we finally start training/evaluating the NN for 30 epochs
train_and_evaluate(cnn_classifier)
# we call the predict method on our estimator
# this method will not return the entire prediction dataset at once but it returns a python generator 
# which we can use to iterate through the predictions one by one
# first we initialize our generator
generator = cnn_classifier.predict(input_fn=predict_input_fn)

# then we store all predictions into a list of dictionaries
# (dictionary from classes and probabilities which we defined in the model function)
predictions = [next(generator) for i in range(len(infer_images))]

# Kaggle are interested only in the classes predictions without the probabilities
# thus we get only the classes in a new list
classes = [predictions[i]['classes'] for i in range(len(predictions))]

# finally we write our predictions into a csv file
def generate_submission_file(predictions, fileName):
    submission = pd.DataFrame()
    
    submission['ImageId'] = range(1,28001,1)
    submission['Label'] = predictions
    submission.set_index('ImageId', inplace=True)
    submission.to_csv(fileName)
    
    print('Submission Ready!')
    
generate_submission_file(classes, 'submission-01.csv')