import numpy as np

# Uninstalling tensorflow 2.0 and downgrading to tensorflow 1.14.0 as tf.log works

# with tf version 1.14. For version 2.0 use tf.math.log.

!pip install 'tensorflow==1.14.0'

import tensorflow as tf
def multiclass_focal_log_loss(y_true, y_pred, class_weights = None, alpha = 0.5, gamma = 2):

    """

    Numpy version of the Focal Loss

    """

    # epsilon 

    eps = 1e-12

    # If actual value is true, keep pt value as y_pred otherwise (1-y_pred)

    pt = np.where(y_true == 1, y_pred, 1-y_pred)

    # If actual value is true, keep alpha_t value as alpha otherwise (1-alpha)

    alpha_t = np.where(y_true == 1, alpha, 1-alpha)

    # Clip values below epsilon and above 1-epsilon

    pt = np.clip(pt, eps, 1-eps)

    # FL = -alpha_t(1-pt)^gamma log(pt)

    focal_loss = -np.mean(np.multiply(np.multiply(alpha_t,np.power(1-pt,gamma)),np.log(pt)), axis=0)

    if class_weights is None:

        focal_loss = np.mean(focal_loss)

    else:

        focal_loss = np.sum(np.multiply(focal_loss, class_weights))

    print(focal_loss)





def get_raw_xentropies(y_true, y_pred):

    y_pred = tf.clip_by_value(y_pred, 1e-7, 1-1e-7)

    xentropies = y_true * tf.log(y_pred) + (1-y_true) * tf.log(1-y_pred)

    return -xentropies



# multilabel focal loss equals multilabel loss in case of alpha=0.5 and gamma=0 

def mutlilabel_focal_loss_inner(y_true, y_pred,class_weights=None, alpha=0.5, gamma=2):

    """

    Tensorflow version of the Focal Loss

    """

    y_true = tf.cast(y_true, tf.float32)

    y_pred = tf.cast(y_pred, tf.float32)



    xentropies = get_raw_xentropies(y_true, y_pred)



    # compute pred_t:

    y_t = tf.where(tf.equal(y_true,1), y_pred, 1.-y_pred)

    alpha_t = tf.where(tf.equal(y_true, 1), alpha * tf.ones_like(y_true), (1-alpha) * tf.ones_like(y_true))



    # compute focal loss contributions

    focal_loss_contributions =  tf.multiply(tf.multiply(tf.pow(1-y_t, gamma), xentropies), alpha_t) 



    # our focal loss contributions have shape (n_samples, s_classes), we need to reduce with mean over samples:

    focal_loss_per_class = tf.reduce_mean(focal_loss_contributions, axis=0)



    # compute the overall loss if class weights are None (equally weighted):

    if class_weights is None:

        focal_loss_result = tf.reduce_mean(focal_loss_per_class)

    else:

        # weight the single class losses and compute the overall loss

        weights = tf.constant(class_weights, dtype=tf.float32)

        focal_loss_result = tf.reduce_sum(tf.multiply(weights, focal_loss_per_class))

    with tf.Session() as sess:

        print(focal_loss_result.eval())
# Dummy matrix to test new function

y_true = np.array([[0,0,0,0,1],[0,0,0,1,0],[0,0,0,0,1]])

y_pred = np.array([[0.22,0.13,0.12,0.90,0.32],[0.11,0.33,0.32,0.45,0.89],[0.32,0.22,0.11,0.16,0.97]])

class_weight = [.5,.15,.15,.1,.1]
%%time

# Numpy version

multiclass_focal_log_loss(y_true, y_pred, class_weights=class_weight)
%%time

# Tensorflow version

mutlilabel_focal_loss_inner(y_true, y_pred, class_weights=class_weight)