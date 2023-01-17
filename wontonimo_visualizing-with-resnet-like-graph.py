%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
import tensorflow as tf
tile_size=(64*2,64*2)
reduction = 2

train_em_image_vol = imread('../input/training.tif')[:40, ::reduction, ::reduction]
train_em_seg_vol = imread('../input/training_groundtruth.tif')[:40, ::reduction, ::reduction]<=0
test_em_image_vol = imread('../input/training.tif')[:40, ::reduction, ::reduction]
test_em_seg_vol = imread('../input/training_groundtruth.tif')[:40, ::reduction, ::reduction]<=0
print("Data Loaded, Dimensions", train_em_image_vol.shape,'->',train_em_seg_vol.shape)
def g_random_tile(em_image_vol, em_seg_vol):
    z_dim, x_dim, y_dim = em_image_vol.shape
    z_pos = np.random.choice(range(z_dim))
    x_pos = np.random.choice(range(x_dim-tile_size[0]))
    y_pos = np.random.choice(range(y_dim-tile_size[1]))
    return np.expand_dims(em_image_vol[z_pos, x_pos:(x_pos+tile_size[0]), y_pos:(y_pos+tile_size[1])],-1), \
            np.expand_dims(em_seg_vol[z_pos, x_pos:(x_pos+tile_size[0]), y_pos:(y_pos+tile_size[1])],-1).astype(float)
np.random.seed(2018)
t_x, t_y = g_random_tile(test_em_image_vol, test_em_seg_vol)
l_x, l_y = g_random_tile(train_em_image_vol, train_em_seg_vol)
print('x:', t_x.shape, 'Range:', t_x.min(), '-', t_x.max())
print('y:', t_y.shape, 'Range:', t_y.min(), '-', t_y.max())
np.random.seed(2016)
t_img, m_img = g_random_tile(test_em_image_vol, test_em_seg_vol)
fig, (ax_img, ax_mask) = plt.subplots(1,2, figsize = (12, 6))
ax_img.imshow(np.clip(255*t_img, 0, 255).astype(np.uint8) if t_img.shape[2]==3 else t_img[:,:,0],
              interpolation = 'none', cmap = 'bone')
ax_mask.imshow(m_img[:,:,0], cmap = 'bone')
xavier = tf.contrib.layers.xavier_initializer_conv2d()

def weight_variable(shape, name="Weight_Variable"):
    return tf.get_variable(initializer=xavier(shape=shape), name=name)

def bias_variable(shape, name="Bias_Variable"):
    return tf.get_variable(initializer=tf.constant(0.1, shape=shape), name=name)

def conv(x, layers_in, layers_out, width=5, stride=1, padding='SAME', name="conv"):
    w = weight_variable([width, width, layers_in, layers_out], name=(name + "_weight"))
    b = bias_variable([layers_out], name=(name + "_bias"))
    return tf.add(tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding=padding), b, name=name)

def batch_normalization(x, training, momentum=0.9, name="none"):
    return tf.layers.batch_normalization(x, training=training, momentum=momentum, name=name + "_norm")

def single_resnet_block(x, layers, width, training, momentum=0.9, name="single_resnet_block"):
    result = batch_normalization(x, training=training, momentum=momentum, name=name)
    result = tf.nn.tanh(result)
    result = conv(result, layers, layers, width=width, name=name)
    return result

def resnet_block(x, layers, width, training, momentum=0.9, name="resnet_block"):
    result = single_resnet_block(x, layers, width, training=training, momentum=momentum, name=(name + "_1"))
    result = single_resnet_block(result, layers, width, training=training, momentum=momentum, name=(name + "_2"))
    return tf.add(x, result, name=name)

def resnet_narrow(x, layers, width, training, narrowing=8, momentum=0.9, name="resnet_narrow"):
    result = batch_normalization(x, training=training, momentum=momentum, name=name)
    result = tf.nn.tanh(result)
    result = conv(result, layers, int(layers / narrowing), width=1, name=name+"_narrowing")
    result = tf.nn.tanh(result)
    result = conv(result, int(layers / narrowing), int(layers / narrowing), width=width, name=name+"_conv")
    result = tf.nn.tanh(result)
    result = conv(result, int(layers / narrowing), layers, width=1, name=name+"_expand")
    result = tf.nn.tanh(result)
    return tf.add(x, result, name=name)

def max_pool(x, stride=2, padding='SAME'):
    return tf.nn.max_pool(x, ksize=[1, stride, stride, 1], strides=[1, stride, stride, 1], padding=padding)

tf.reset_default_graph()

layers_of_depth = 32
conv_width = 17
x_min = t_x.min()
x_max = t_x.max()

# Inputs
x_in = tf.placeholder(tf.float32, [None, None, None], name="x0")
y_in = tf.placeholder(tf.float32, [None, None, None], name="y0")
is_training = tf.placeholder(tf.bool)

x_normalize = (x_in-x_min) / (x_max-x_min)
x_in_reshaped = tf.expand_dims( x_normalize, -1) # reshape to be a tensor of shape [batch, height, width, 1]

# Induction - define the graph
x_to_layers = tf.nn.tanh(conv(x=x_in_reshaped, layers_in=1, layers_out=layers_of_depth, width=conv_width, name="to_layers"))
conv_layer = x_to_layers
for layer in range(4):
    conv_layer = resnet_narrow(conv_layer, layers=layers_of_depth, width=conv_width, training=is_training, name="res_block_"+str(layer))

y_out_large = conv(x=conv_layer, layers_in=layers_of_depth, layers_out=1, width=conv_width, name="to_output")
y_out_logits = tf.squeeze( y_out_large ) # reshape to be a tensor of shape [batch, height, width]
y_out = tf.sigmoid( y_out_logits )

# Loss - use cross entropy
# loss1 =  tf.reduce_sum( y_in * tf.nn.sigmoid_cross_entropy_with_logits(labels=y_in, logits=y_out_logits, name="loss1") ) / tf.reduce_sum(y_in)
# loss0 =  tf.reduce_sum( (1.0 - y_in) * tf.nn.sigmoid_cross_entropy_with_logits(labels=y_in, logits=y_out_logits, name="loss0") ) / tf.reduce_sum(1. - y_in)
loss1 = -1 * tf.reduce_sum( y_in * tf.log(y_out) ) / tf.reduce_sum(y_in)
loss0 = -1 * tf.reduce_sum( (1. - y_in) * tf.log(1. - y_out) ) / tf.reduce_sum(1. - y_out)


#loss0 = tf.reduce_sum( (1. - y_in) * tf.square(y_in-y_out) ) / tf.reduce_sum(1. - y_in) + 0.02
#loss1 = tf.reduce_sum( y_in * tf.square(y_in-y_out) ) / tf.reduce_sum(y_in) + 0.02

# Train
learning_rate = tf.placeholder_with_default(1e-4,shape=[])
train = tf.train.AdamOptimizer(learning_rate).minimize( loss0 + loss1)
# create a session
sess = tf.Session()
# initialize session variables
sess.run(tf.global_variables_initializer())

def show_graph_examples(seed=2016) :
    np.random.seed(seed)
    t_img, m_img = g_random_tile(test_em_image_vol, test_em_seg_vol)
    example_out = sess.run(y_out, feed_dict={x_in:t_img.reshape([1] + list(tile_size)), is_training:False}).reshape(list(tile_size) + [1])
    fig, (ax_img, ax_mask, ax_pred) = plt.subplots(1,3, figsize = (12, 6))
    ax_img.imshow(np.clip(255*t_img, 0, 255).astype(np.uint8) if t_img.shape[2]==3 else t_img[:,:,0],
                  interpolation = 'none', cmap = 'bone')
    ax_mask.imshow(m_img[:,:,0], cmap = 'bone')
    ax_pred.imshow((example_out[:,:,0]-.5)*2., cmap = 'bone')
    
    print(example_out.min() , example_out.max() , example_out.mean() )
    print(m_img.min() , m_img.max() , m_img.mean() )

show_graph_examples()
sess.run(tf.global_variables_initializer())
loss_val0, loss_val1 = sess.run((loss0,loss1), feed_dict={x_in:t_img, y_in:m_img, is_training:True})
print (loss_val0.mean(),loss_val1.mean())
rows = test_em_image_vol.shape[0]
def train_once_and_visualize() :
    for step in range(0,rows,2):
        _, loss_val0,loss_val1 = sess.run((train,loss0,loss1), feed_dict={x_in:test_em_image_vol[ (step%rows) : (step%rows+2) ,:,:],
                                y_in:test_em_seg_vol[ (step%rows) : (step%rows+2) ,:,:], 
                                is_training:True, learning_rate:1e-5})
    print ("step",step,", loss",loss_val0.mean(),loss_val1.mean(),loss_val0.mean()+loss_val1.mean())
    show_graph_examples()
train_once_and_visualize()
train_once_and_visualize()
train_once_and_visualize()
train_once_and_visualize()