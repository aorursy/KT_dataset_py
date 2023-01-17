import tensorflow as tf

from gensim.models import word2vec

from gensim.models import Word2Vec

import pandas as pd

import glob

import numpy as np

import os

import matplotlib.pyplot as plt

%matplotlib inline
os.listdir('../input/')
os.listdir('../input/102flowersdataset/')
os.listdir('../input/cvpr2016/cvpr2016_flowers')
n_input = 100

n_hidden = 128

image_height = 64

image_width = 64

image_depth = 3

noise_dim = 100

maxlength = 250

NUM_EPOCHS = 100

batch_size = 64
# 将图片解压至102flowers文件夹(当前运行目录下创建)

if not os.path.exists('102flowers'):

    !mkdir 102flowers

    !tar zxvf ../input/102flowersdataset/102flowers.tgz -C ./102flowers/

    print()

# 取出所有描述花的TXT文本

all_text_filename = glob.glob('../input/cvpr2016/cvpr2016_flowers/text_c10/class_*/image_*.txt')

# 对文本编号进行排序，为了方便使文本和图片对应起来

all_text_filename.sort(key=lambda x:x.split('/')[-1])



print(all_text_filename[:2])
# 导入图片并排序

all_image_filename = glob.glob('./102flowers/jpg/*.jpg')

all_image_filename.sort()
all_image_filename[3001]
all_text_filename[3001]
# 输入队列需要两部分，首先是图片和对应文本的正确输入，然后是图片和文本不对应的错误输入

all_text_filename = np.array(all_text_filename)

all_image_filename = np.array(all_image_filename)

# 乱序排列图片制造错误输入

wrong_image_filename = all_image_filename[np.random.permutation(len(all_image_filename))]
dataset_image = tf.data.Dataset.from_tensor_slices((all_image_filename, wrong_image_filename))
dataset_image
# 建一个all_text文件存放所有文本

if not os.path.exists('../input/text_to_image/all_text.txt'):

    with open('all_text.txt', 'at') as f:

        # 把每一个文本的换行去掉，只在队尾加换行，然后写入all_text文件

        for a_text in all_text_filename:

            f.write(open(a_text).read().replace('\n', '') + '\n')

if not os.path.exists('../input/text_to_image/word_model'):

    # 用word2vec.Text8Corpus方法训练文本文件

    sentences = word2vec.Text8Corpus('all_text.txt')

    # 将每个词转换为长度100的向量

    model = word2vec.Word2Vec(sentences, size=100)

    # 每一次训练结果都有随机性，这里把第一次训练结果保存下来，保证每一次训练的词向量都是相同的

    model.save('word_model')

else:

    model = Word2Vec.load('../input/text_to_image/word_model')

    !cp ../input/text_to_image/all_text.txt ./

    !cp ../input/text_to_image/word_model ./

word_vectors = model.wv
# 获取最长文本的长度作为RNN中每一个输入队列的长度

maxlength = max([len(open(a_text).read().split()) for a_text in all_text_filename])
n_steps = maxlength
# 向量填充

def pad(x, maxlength=200):

    x1 = np.zeros((maxlength,100))

    x1[:len(x)] = x

    return x1
# 把所有文本向量化

def text_vec(text_filenames):

    vec = []

    for a_text in text_filenames:

        # 读取文本并切分单词

        all_word = open(a_text).read().split()

        # 把每一个单词转换成词向量

        all_vec = [word_vectors[w] for w in all_word if w in word_vectors]

        vec.append(all_vec)

    data = pd.Series(vec)

    # 向量填充

    data = data.apply(pad, maxlength=maxlength)

    # 将所有向量连接成一个矩阵 8189*223*100

    data_ = np.concatenate(data).reshape(len(data),maxlength,100)

    return data_
data_text_emb = text_vec(all_text_filename)
data_text_emb.shape
# 图片处理函数

def read_image(image_filename):

    image = tf.read_file(image_filename)

    image = tf.image.decode_jpeg(image, channels=3)

    # 转换成512*512大小，然后转换成256*256大小

    image = tf.image.resize_image_with_crop_or_pad(image, 512, 512)

    image = tf.image.resize_images(image, (256, 256))

    #image = tf.image.convert_image_dtype(image, tf.float32)

    # 将图片0-1标准化

    image = (image - tf.reduce_min(image))/(tf.reduce_max(image) - tf.reduce_min(image))

    return image
# 读取图片

def _pre_func(real_image_name, wrong_image_name):

    wrong_image = read_image(wrong_image_name)

    real_image = read_image(real_image_name)

    return real_image, wrong_image
# 图片输入队列

dataset_image = dataset_image.map(_pre_func)
dataset_image = dataset_image.batch(batch_size)
# 通过Iterator方法得到每一个batch的数据

iterator = tf.data.Iterator.from_structure(dataset_image.output_types, dataset_image.output_shapes)

real_image_batch, wrong_image_batch = iterator.get_next()
input_text = tf.placeholder(tf.float32, [None, n_steps, n_input])

inputs_noise = tf.placeholder(tf.float32, [None, noise_dim], name='inputs_noise')
# 求文本长度

def length(shuru):

    return tf.reduce_sum(tf.sign(tf.reduce_max(tf.abs(shuru),reduction_indices=2)),reduction_indices=1)
def text_rnn(input_text, batch_size=64, reuse=None):

    cell = tf.contrib.rnn.GRUCell(n_hidden,

                                  kernel_initializer = tf.truncated_normal_initializer(stddev=0.0001),

                                  bias_initializer = tf.truncated_normal_initializer(stddev=0.0001),

                                  reuse=reuse)

    output, _ = tf.nn.dynamic_rnn(

                                  cell,

                                  input_text,

                                  dtype=tf.float32,

                                  sequence_length = length(input_text)

                                  )



    # 取出一个batch中每一个文本对应的output

    index = tf.range(0,batch_size)*n_steps + (tf.cast(length(input_text),tf.int32) - 1)

    flat = tf.reshape(output,[-1,int(output.get_shape()[2])])

    last = tf.gather(flat,index)

    return last
def get_generator(noise_img, image_depth, condition_label, is_train=True, alpha=0.2):

    with tf.variable_scope("generator", reuse=(not is_train)):

        # 100 x 1 to 4 x 4 x 512

        # 全连接层

        noise_img = tf.to_float(noise_img)

        noise_img = tf.layers.dense(noise_img, n_hidden)

        noise_img = tf.maximum(alpha * noise_img, noise_img)

        noise_img_ = tf.concat([noise_img, condition_label], 1)

        # 全连接

        layer1 = tf.layers.dense(noise_img_, 4*4*512)

        layer1 = tf.reshape(layer1, [-1, 4, 4, 512])

        layer1 = tf.layers.batch_normalization(layer1, training=is_train)

        layer1 = tf.nn.relu(layer1)

        # batch normalization

        #layer1 = tf.layers.batch_normalization(layer1, training=is_train)

        # ReLU

        #layer1 = tf.nn.relu(layer1)

        # dropout

        layer1 = tf.nn.dropout(layer1, keep_prob=0.8)

        

        # 反卷积

        # 4 x 4 x 512 to 8 x 8 x 256

        layer2 = tf.layers.conv2d_transpose(layer1, 256, 3, strides=2, padding='same')

        layer2 = tf.layers.batch_normalization(layer2, training=is_train)

        layer2 = tf.nn.relu(layer2)

        layer2 = tf.nn.dropout(layer2, keep_prob=0.8)

        

        # 8 x 8 256 to 16x 16 x 128

        layer3 = tf.layers.conv2d_transpose(layer2, 128, 3, strides=2, padding='same')

        layer3 = tf.layers.batch_normalization(layer3, training=is_train)

        layer3 = tf.nn.relu(layer3)

        layer3 = tf.nn.dropout(layer3, keep_prob=0.8)

        

        # 16 x 16 x 128 to 32 x 32 x 64

        layer4 = tf.layers.conv2d_transpose(layer3, 64, 3, strides=2, padding='same')

        layer4 = tf.layers.batch_normalization(layer4, training=is_train)

        layer4 = tf.nn.relu(layer4)

        

        # 64 x 64 x 32

        layer5 = tf.layers.conv2d_transpose(layer4, 32, 3, strides=2, padding='same')

        layer5 = tf.layers.batch_normalization(layer5, training=is_train)

        layer5 = tf.nn.relu(layer5)

        

        # 128 x 128 x 16

        layer6 = tf.layers.conv2d_transpose(layer5, 16, 3, strides=2, padding='same')

        layer6 = tf.layers.batch_normalization(layer6, training=is_train)

        layer6 = tf.nn.relu(layer6)  

        

        #  256 x 256 x 3

        logits = tf.layers.conv2d_transpose(layer6, image_depth, 3, strides=2, padding='same')

        outputs = tf.tanh(logits)

        outputs = (outputs/2) + 0.5

        # 将输出规范到0,1之间，防止很小的数/2之后出错

        outputs = tf.clip_by_value(outputs, 0.0, 1.0)

        return outputs
def get_discriminator(inputs_img, condition_label, reuse=False, alpha=0.2):

    with tf.variable_scope("discriminator", reuse=reuse):

        # 256 x 256 x 3 to 128 x 128 x 16

        # 第一层不加入BN

        layer1 = tf.layers.conv2d(inputs_img, 16, 3, strides=2, padding='same')

        layer1 = tf.maximum(alpha * layer1, layer1)

        layer1 = tf.nn.dropout(layer1, keep_prob=0.8)

        

        # 128 x 128 x 16 to 64 x 64 x 32

        layer2 = tf.layers.conv2d(layer1, 32, 3, strides=2, padding='same')

        layer2 = tf.layers.batch_normalization(layer2, training=True)

        layer2 = tf.maximum(alpha * layer2, layer2)

        layer2 = tf.nn.dropout(layer2, keep_prob=0.8)

        

        # 32 x 32 x 64

        layer3 = tf.layers.conv2d(layer2, 64, 3, strides=2, padding='same')

        layer3 = tf.layers.batch_normalization(layer3, training=True)

        layer3 = tf.maximum(alpha * layer3, layer3)

        layer3 = tf.nn.dropout(layer3, keep_prob=0.8)

        

        # 16*16*128

        layer4 = tf.layers.conv2d(layer3, 128, 3, strides=2, padding='same')

        layer4 = tf.layers.batch_normalization(layer4, training=True)

        layer4 = tf.maximum(alpha * layer4, layer4)

        

        

         # 8*8*256

        layer5 = tf.layers.conv2d(layer4, 256, 3, strides=2, padding='same')

        layer5 = tf.layers.batch_normalization(layer5, training=True)

        layer5 = tf.maximum(alpha * layer5, layer5)

        

         # 4*4*512

        layer6 = tf.layers.conv2d(layer5, 512, 3, strides=2, padding='same')

        layer6 = tf.layers.batch_normalization(layer6, training=True)

        layer6 = tf.maximum(alpha * layer6, layer6)

        

    

        

        text_emb = tf.layers.dense(condition_label, 512)

        text_emb = tf.maximum(alpha * text_emb, text_emb)

        text_emb = tf.expand_dims(text_emb, 1)

        text_emb = tf.expand_dims(text_emb, 2)

        # tf.tile是按照给定情况复制张量

        text_emb = tf.tile(text_emb, [1,4,4,1])

        # 将图像和文本条件结合在一起

        layer_concat = tf.concat([layer6, text_emb], 3)

        

        layer7 = tf.layers.conv2d(layer_concat, 512, 1, strides=1, padding='same')

        layer7 = tf.layers.batch_normalization(layer7, training=True)

        layer7 = tf.maximum(alpha * layer7, layer7)

        

        flatten = tf.reshape(layer7, (-1, 4*4*512))

        logits = tf.layers.dense(flatten, 1)

        outputs = tf.sigmoid(logits)

        

        return logits, outputs
def get_loss(inputs_image, wrong_image, inputs_noise, condition_label, image_depth, smooth=0.1):

    g_outputs = get_generator(inputs_noise, image_depth, condition_label, is_train=True)

    d_logits_real, d_outputs_real = get_discriminator(inputs_image, condition_label)

    d_logits_fake, d_outputs_fake = get_discriminator(g_outputs, condition_label, reuse=True)

    d_logits_wrong, d_outputs_wrong = get_discriminator(wrong_image, condition_label, reuse=True)

    

    print(inputs_image.get_shape(), condition_label.get_shape())

    

    # 计算Loss

    # 生成器希望生成的图片能被判定为接近1

    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, 

                                                                    labels=tf.ones_like(d_outputs_fake)*(1-smooth)))

    

    #g_loss_l1 = tf.reduce_mean(tf.abs(g_outputs - inputs_image))

    

    #g_loss = g_loss_ + g_loss_l1

    # 判别器希望：真实图片判定为接近1，生成图片和与文本描述不符的图片判定为接近0

    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,

                                                                         labels=tf.ones_like(d_outputs_real)*(1-smooth)))

    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,

                                                                         labels=tf.ones_like(d_outputs_fake)*smooth))

    d_loss_wrong = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_wrong,

                                                                         labels=tf.ones_like(d_outputs_wrong)*smooth))

    

    d_loss = d_loss_real + d_loss_fake + d_loss_wrong

    

    return g_loss, d_loss
# 梯度下降优化器

def get_optimizer(g_loss, d_loss, beta1=0.4, learning_rate=0.001):

    train_vars = tf.trainable_variables()

    

    g_vars = [var for var in train_vars if var.name.startswith("generator")]

    d_vars = [var for var in train_vars if var.name.startswith("discriminator")]

    

    # Optimizer

    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):

        g_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)

        d_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)

    

    return g_opt, d_opt
# 画图函数

def plot_images(samples):

    #samples = (samples+1)/2

    fig, axes = plt.subplots(nrows=1, ncols=10, sharex=True, sharey=True, figsize=(20,2))

    for img, ax in zip(samples, axes):

        ax.imshow(img.reshape((256, 256, 3)))

        ax.get_xaxis().set_visible(False)

        ax.get_yaxis().set_visible(False)

    fig.tight_layout(pad=0)
# 生成图片效果展示函数

def show_generator_output(sess, n_images, inputs_noise, output_dim, test_text_vec):

#    condition_text = tf.to_float(condition_text)

#    last, b_size = sess.run(text_vec(condition_text, batch_size=n_images, reuse=True))

    samples = sess.run(get_generator(inputs_noise, output_dim, test_text_vec, is_train=False))

    return samples
# 定义参数

n_samples = 10

learning_rate = 0.0002

beta1 = 0.5
losses = []

step = 0

last = text_rnn(input_text)

g_loss, d_loss = get_loss(real_image_batch, wrong_image_batch, inputs_noise, last, image_depth, smooth=0.1)

g_train_opt, d_train_opt = get_optimizer(g_loss, d_loss, beta1, learning_rate)

saver = tf.train.Saver()
  
    
show = 1

if show:

    # 展示前五张图片和前五个文本描述

    for a_text in all_text_filename[:5]:

            # 读取文本并切分单词

            all_word = open(a_text).read()

            print(all_word)



    with tf.Session() as sess:

    #     sess.run(tf.global_variables_initializer())

        # 恢复检查点模型变量

        model_file=tf.train.latest_checkpoint('../input/text-to-iamge')

        saver.restore(sess, model_file)

        n_samples = 5

        # 取出前几个描述看看生成效果

        condition_text = data_text_emb[:n_samples]

        print

         # 噪声

        test_noise = np.random.uniform(-1, 1, size=[n_samples, noise_dim])

        # 提取文本特征

        last_test = text_rnn(input_text, batch_size=n_samples, reuse=True)

        test_text_vec = sess.run(last_test, feed_dict={input_text: condition_text})

        samples = show_generator_output(sess, n_samples, test_noise, 3, test_text_vec)

        plot_images(samples)  

    

else:

    # 训练过程

    saver = tf.train.Saver()

    with tf.Session() as sess:

    #     sess.run(tf.global_variables_initializer())

        # 恢复检查点模型变量

        model_file=tf.train.latest_checkpoint('../input/text-to-iamge')

        saver.restore(sess, model_file)



        for epoch in range(560, 571):

            # 首先对所有数据乱序处理，这时图片和文本是对应的

            index = np.random.permutation(len(all_image_filename))

            data_text_emb = data_text_emb[index]

            all_image_filename = all_image_filename[index]

            # 制造错误输入：将图片乱序，文本序号不变

            wrong_image_filename = all_image_filename[np.random.permutation(len(all_image_filename))] 

            dataset_image = tf.data.Dataset.from_tensor_slices((all_image_filename, wrong_image_filename))

            dataset_image = dataset_image.map(_pre_func)

            # 在每个epoch里面每张图片出现一次

            dataset_image = dataset_image.repeat(1)

            dataset_image = dataset_image.batch(batch_size)

            # 用乱序处理之后的数据集初始化iterator

            dataset_image_op = iterator.make_initializer(dataset_image)



            sess.run(dataset_image_op)

            i = 0

            while True: 



                try:

                    batch_noise = np.random.uniform(-1, 1, size=(batch_size, noise_dim))

                    text_emb_batch = data_text_emb[i: i + batch_size]

                    i = i + batch_size

                    _ = sess.run([g_train_opt, d_train_opt], feed_dict={input_text: text_emb_batch,

                                                                inputs_noise: batch_noise})



    #               if step % 50 == 0:

    #                   saver.save(sess, "./model10.ckpt")

    #                   train_loss_d = d_loss.eval({input_text: text_emb_batch,

    #                                               inputs_noise: batch_noise})

    #                   train_loss_g = g_loss.eval({input_text: text_emb_batch,

    #                                               inputs_noise: batch_noise})

    #                   

    #                   losses.append((train_loss_d, train_loss_g))

    #                   print("Step {}....".format(step+1), 

    #                         "Discriminator Loss: {:.4f}....".format(train_loss_d),

    #                         "Generator Loss: {:.4f}....". format(train_loss_g))







                        # 显示图片

                    step += 1    

                #except tf.errors.OutOfRangeError as e:

                # 最后一个batch不够batchsize,打印epoch终止提示信息

                except:

    #                 saver.save(sess, "./model10.ckpt")

                    print('epoch', epoch, 'step', step)

                    #print(e)

                    #try:

                    #    sess.run(real_image_batch)

                    #except Exception as e:

                    #    print(e)

                    break



            if epoch%2 == 0:

                #saver.save(sess, "./model10.ckpt")

                n_samples = 10

                # 取出前10个描述看看生成效果

                condition_text = data_text_emb[:n_samples]

                # 噪声

                test_noise = np.random.uniform(-1, 1, size=[n_samples, noise_dim])

                # 提取文本特征

                last_test = text_rnn(input_text, batch_size=n_samples, reuse=True)

                test_text_vec = sess.run(last_test, feed_dict={input_text: condition_text})

                samples = show_generator_output(sess, n_samples, test_noise, 3, test_text_vec)

                plot_images(samples)

        saver.save(sess, "./model11.ckpt")
!rm -rf 102flowers