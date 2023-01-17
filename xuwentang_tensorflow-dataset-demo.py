# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)



# **限制能够使用的GPU资源**

session_config = tf.ConfigProto()

session_config.gpu_options.allow_growth = True
!nvidia-smi
ds = tf.data.Dataset.range(5)

iterator = ds.make_one_shot_iterator()

batch_xs = iterator.get_next()

print(batch_xs)



with tf.Session(config=session_config) as sess:

    try:

        while True:

            print(sess.run(batch_xs))

    except tf.errors.OutOfRangeError:

        print('Done reading')

    except KeyboardInterrupt:

        print('Stopped by the user')
!nvidia-smi
# 训练数据

train_x = [

    [1.2, 2.0, 3.1, 1.4],

    [4.1, 2.3, 1.9, 2.0],

    [1.9, 7.1, 2.6, 1.2]

]

train_y = [4.0, 1.0, 2.0]







def my_input_fn(features, labels, perform_shuffle=False, batch_size=1, repeat_count=1):

    # Construct the dataset object

    dataset_features = tf.data.Dataset.from_tensor_slices(features)

    dataset_labels = tf.data.Dataset.from_tensor_slices(labels)

    dataset = tf.data.Dataset.zip((dataset_features, dataset_labels))

    # Shuffle

    if perform_shuffle:

        dataset = dataset.shuffle(1000)

    # repeat, batch

    dataset = dataset.repeat(repeat_count).batch(batch_size)

    iterator = dataset.make_one_shot_iterator()

    batch_xs, batch_ys = iterator.get_next()

    return batch_xs, batch_ys





def print_iterator_get_next():

    with tf.Session(config=session_config) as sess:

        try:

            for i in range(100):

                batch_xs_, batch_ys_ = sess.run([batch_xs, batch_ys])

                print('---第%(num)d次解析数据---' % {'num': i + 1})

                print('input_features:', batch_xs_, ', input_labels', batch_ys_)

        except tf.errors.OutOfRangeError:

            print('Done reading')

        except KeyboardInterrupt:

            print('Stopped by the user')





batch_xs, batch_ys = my_input_fn(train_x, train_y, batch_size=2, repeat_count=1)

# print_iterator_get_next()
x = tf.placeholder(dtype=tf.float32, shape=(None, 4), name='x')

y = tf.placeholder(dtype=tf.float32, shape=(None), name='y')



W = tf.Variable(tf.zeros(shape=(4, 1)), name='weight')

b = tf.Variable(tf.zeros(shape=(1)), name='bais')

# 线性回归

preds = tf.matmul(x, W) + b



loss = tf.losses.mean_squared_error(y, preds)

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)



# 变量初始化器

init = tf.global_variables_initializer()

with tf.Session(config=session_config) as sess:

    sess.run(init)

    

    trainning_steps = 2

    for step in range(trainning_steps):

        # 先求该批数据的值

        batch_xs_, batch_ys_ = sess.run([batch_xs, batch_ys])

        # 用求得的值，迭代优化模型

        _, trian_loss = sess.run([train_step, loss], 

                                 feed_dict={

                                     x: batch_xs_,

                                     y: batch_ys_

                                 })

        print('Step: %d' % (step + 1), ' loss:{:.9f}'.format(trian_loss))
def my_model(features, labels, mode, params):

    '''自定义模型函数

    '''

    W = tf.Variable(tf.zeros(shape=(4, 1)), name='weight')

    b = tf.Variable(tf.zeros(shape=(1)), name='bais')

    predictions = tf.matmul(features, W) + b

    

    ''' 预测返回 '''

    if mode == tf.estimator.ModeKeys.PREDICT:

        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    

    # 定义损失函数

    loss = tf.losses.mean_squared_error(labels=tf.reshape(labels, shape=(-1, 1)), 

                                        predictions=predictions)

    # 添加评估输出项

    meanloss = tf.metrics.mean(loss)

    metrics = {'meanloss': meanloss}

    ''' 评估（测试）返回 '''

    if mode == tf.estimator.ModeKeys.EVAL:

        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    

    # 训练处理

    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdagradOptimizer(learning_rate=params['learning_rate'])

    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)





# 创建 tf.estimator.Estimator 对象

estimator = tf.estimator.Estimator(model_fn=my_model, 

                                   model_dir='./model',

                                   config=tf.estimator.RunConfig(session_config=session_config),

                                   params={

                                       'learning_rate': 0.1,

                                   })



estimator.train(lambda: my_input_fn(train_x, train_y, batch_size=2, repeat_count=1), steps=10)

tf.logging.info('训练完成')
!ls model
a = np.array([1, 2, 3])

b = np.array([2, 5, 2])



a_2 = np.array(

    [

        [1, 2, 3],

        [4, 5, 6]

    ]

)

b_2 = np.array(

    [

        [2],

        [5],

        [2]

    ]

)

print('a:', a)

print('b:', b)

print('a_2:', a_2)

print('b_2:', b_2)



with tf.Session(config=session_config) as sess:

    # multiply 形状必须相同（扩展后），按位乘

    print('a*b:', sess.run(tf.multiply(a, b)))

    # matmul, 矩阵点乘，2x3, 3x1 -> 2x1

    print('a.dot(b):', sess.run(tf.matmul(a_2, b_2)))
loss = tf.losses.mean_squared_error([2, 3, 1, 2], [0, 0, 0, 0])



sess = tf.Session(config=session_config)

sess.run(tf.global_variables_initializer())

sess.run(tf.local_variables_initializer())

sess.run(loss)

sess.close()
labels = np.array([2, 1, 0, 2])

preds = np.array([2, 0, 0, 2])



print(labels == preds)

print((labels == preds).mean())
input_predictions = tf.placeholder(dtype=tf.int32, shape=(None), name='preds')

input_labels = tf.placeholder(dtype=tf.int32, shape=(None), name='labels')



correct_predictions = tf.equal(input_predictions, input_labels)

accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))



with tf.Session(config=session_config) as sess:

    sess.run(tf.global_variables_initializer())

    _accuracy = sess.run(accuracy, feed_dict={

        input_predictions: preds,

        input_labels: labels

    })

    print('accuracy:', _accuracy)

x = tf.constant([

    [1., 1.], 

    [2., 2.]

])



with tf.Session(config=session_config) as sess:

    print(sess.run(tf.reduce_mean(x)))  # 1.5

    # 默认，reduce_mean 一下，秩就减小1

    print(sess.run(tf.reduce_mean(x, axis=0)))  # [1.5, 1.5]

    # keep_dims=True, 保持矩阵的秩不变

    print(sess.run(tf.reduce_mean(x, axis=0, keep_dims=True)))  # [[1.5 1.5]]

    print(sess.run(tf.reduce_mean(x, axis=1)))  # [1.,  2.]