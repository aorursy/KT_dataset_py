import os 

import time

import imageio

import numpy as np 

import tensorflow as tf 

import matplotlib.pyplot as plt 

%matplotlib inline
HR_IMAGE_SIZE = (128,128)

LR_IMAGE_SIZE = (32,32)
IMG_PATH = "/kaggle/input/cocotest2014/test2014/"

PATH_PATTERN = "/kaggle/input/cocotest2014/test2014/*.jpg"

INIT_INFER_OUTPUT_PATH = "/kaggle/working/G_init_output/"

INFER_OUTPUT_PATH = "/kaggle/working/img_output/"

target_img = os.path.join(IMG_PATH,"COCO_test2014_000000000001.jpg")
img = plt.imread(target_img)

plt.axis("off")

plt.imshow(img)
def filter_func(filename):

    # 过滤掉灰度图像和分辨率小于256的图像，只保留分辨率大于256的rgb图像

    img = tf.io.read_file(filename)

    shape = tf.io.extract_jpeg_shape(img)

    return shape[-1] == 3 and shape[0] >= 128 and shape[1] >= 128



def parse_image(filename):

    image = tf.io.read_file(filename)

    image = tf.image.decode_jpeg(image)

    image = tf.image.convert_image_dtype(image,tf.float32) # 数据类型转换成float之后,范围变成了[0,1]

    hr_img = tf.image.random_crop(image, (*HR_IMAGE_SIZE,3))

    # hr_img = tf.image.resize(image,HR_IMAGE_SIZE)

    hr_img = hr_img * 2.0 - 1.0  

    lr_img = tf.image.resize(hr_img, LR_IMAGE_SIZE)

    return hr_img, lr_img



def get_data_gen(path_pattern=PATH_PATTERN,batch_size=64,pref = 3):

    dataset = tf.data.Dataset.list_files(path_pattern).filter(filter_func)

    dataset = dataset.map(parse_image,num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.shuffle(2).batch(batch_size).prefetch(buffer_size=pref)

    return dataset
def load_img(path_to_img, target_size, offset=(200,100)):



    # 读出来是字节流，需要解码

    img = tf.io.read_file(path_to_img)

    img = tf.image.decode_image(img, channels=3)

    # uint8->float32

    img = tf.image.convert_image_dtype(img, tf.float32)

#     img = tf.image.random_crop(img, (*HR_IMAGE_SIZE,3))

    img = tf.image.crop_to_bounding_box(img,offset[0],offset[1],

                                        HR_IMAGE_SIZE[0],HR_IMAGE_SIZE[1])

    img = tf.image.resize(img, target_size)

    return img



def generateGIF(imgs_path, gif_name, duration=0.5):

    frames = []

    img_name_list = os.listdir(imgs_path)

    for img_name in img_name_list:

        img_path = os.path.join(imgs_path, img_name)

        img = imageio.imread(img_path)

        frames.append(img)

    imageio.mimsave(gif_name, frames, "GIF", duration=duration)





def plot_loss(G_loss, D_loss):

    plt.figure()

    plt.plot(range(len(G_loss)), G_loss, label="G_loss")

    plt.plot(range(len(D_loss)), D_loss, label="D_loss")

    plt.legend()

    plt.grid()

    plt.savefig("loss.png")
def vgg_extractor(output_layer="block4_conv4"):



#     weights = "vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5"

    vgg = tf.keras.applications.VGG19(include_top=False, weights="imagenet")

    vgg.trainable = False

    output = vgg.get_layer(output_layer).output

    model = tf.keras.Model([vgg.input], [output])

    return model
class ResidualBlock(tf.keras.layers.Layer):



    def __init__(self, num_filters, filter_size=3, strides=1, padding="SAME", identity=True):

        super(ResidualBlock, self).__init__()

        self.identity = identity



        self.conv1 = tf.keras.layers.Conv2D(

            filters=num_filters, kernel_size=filter_size, strides=strides, padding=padding)

        self.bn1 = tf.keras.layers.BatchNormalization()

        self.prl1 = tf.keras.layers.PReLU()



        self.conv2 = tf.keras.layers.Conv2D(

            filters=num_filters, kernel_size=filter_size, strides=strides, padding=padding)

        self.bn2 = tf.keras.layers.BatchNormalization()



        if not identity:

            self.conv3 = tf.keras.layers.Conv2D(

                filters=num_filters, kernel_size=1, strides=1, padding=padding)



        self.add = tf.keras.layers.add



    def call(self, x, training=True):

        y = self.conv1(x)

        y = self.bn1(y, training)

        y = self.prl1(y)



        y = self.conv2(y)

        y = self.bn2(y, training)

        if not self.identity:

            x = self.conv3(x)



        return self.add([y, x])





class PiexlShuffle2D(tf.keras.layers.Layer):



    def __init__(self, upsampling_factor=2, data_format="channels_last", **kwargs):

        super(PiexlShuffle2D, self).__init__(**kwargs)

        self.upsampling_factor = upsampling_factor

        self.data_format = data_format



    def build(self, input_shape):

        factor = self.upsampling_factor ** 2

        if(self.data_format not in ("channels_last", "channels_first")):

            raise ValueError(

                "The data_format should be 'channels_last' or 'channels_first'.")



        if ((self.data_format == "channels_last") and (input_shape[-1] % factor != 0)):

            # [N,H,W,C]

            raise ValueError(

                "The number of channels should be of integer times of upsampling_factor^2. ")

        elif ((self.data_format == "channels_first") and (input_shape[1] % factor != 0)):

            raise ValueError(

                "The number of channels should be of integer times of upsampling_factor^2. ")



        super(PiexlShuffle2D, self).build(input_shape)



    def call(self, x):

        if self.data_format == "channels_first":

            data_format = "NCHW"

        elif self.data_format == "channels_last":

            data_format = "NHWC"

        return tf.nn.depth_to_space(x, self.upsampling_factor, data_format=data_format)



    def compute_output_shape(self, input_shape):



        if self.data_format == "channels_last":

            N, H, W, C = input_shape

            return (N, H * self.upsampling_factor, W * self.upsampling_factor, C / self.upsampling_factor ** 2)

        else:

            N, C, H, W = input_shape

            return (N, C / self.upsampling_factor, H * self.upsampling_factor, W * self.upsampling_factor)





class SubPixelConv2D(tf.keras.layers.Layer):



    def __init__(self, num_filters, kernel_size=3, strides=1, padding="same", upsampling_factor=2):

        super(SubPixelConv2D, self).__init__()



        self.conv = tf.keras.layers.Conv2D(

            num_filters, kernel_size, strides, padding=padding)

        self.psf = PiexlShuffle2D(upsampling_factor)

        self.prl = tf.keras.layers.PReLU()



    def call(self, x):

        y = self.conv(x)

        y = self.psf(y)

        y = self.prl(y)

        return y





class SISRGenerator(tf.keras.Model):



    def __init__(self, num_resblock):

        super(SISRGenerator, self).__init__()



        # self.inputs = tf.keras.layers.InputLayer(input_shape)

        self.conv1 = tf.keras.layers.Conv2D(64, 9, 1, padding="SAME")

        self.prl = tf.keras.layers.PReLU()



        self.rb_list = [ResidualBlock(

            64, 3, 1) for i in range(num_resblock)]



        self.conv2 = tf.keras.layers.Conv2D(64, 3, 1, padding="SAME")

        self.bn1 = tf.keras.layers.BatchNormalization()



        self.add = tf.keras.layers.add



        self.spc1 = SubPixelConv2D(256, 3, 1, upsampling_factor=2)

        self.spc2 = SubPixelConv2D(256, 3, 1, upsampling_factor=2)



        self.conv3 = tf.keras.layers.Conv2D(3, 3, 1, padding="same")



    def call(self, x, training=True):

        y = self.conv1(x)

        y = self.prl(y)



        for rb in self.rb_list:

            y = rb(y, training)

        short_cut = y

        y = self.conv2(y)

        y = self.bn1(y, training)

        y = self.add([y, short_cut])



        y = self.spc1(y)

        y = self.spc2(y)



        y = self.conv3(y)

        return y

class ConvBlock(tf.keras.layers.Layer):



    def __init__(self, num_filters=64, filter_size=3, strides=1, padding="SAME", bn_flag=True):

        super(ConvBlock, self).__init__()

        self.bn_flag = bn_flag

        self.conv = tf.keras.layers.Conv2D(

            num_filters, filter_size, strides, padding=padding)

        if bn_flag:

            self.bn = tf.keras.layers.BatchNormalization()

        self.lrl = tf.keras.layers.LeakyReLU(alpha=0.3)



    def call(self, x):

        y = self.conv(x)

        if self.bn_flag:

            y = self.bn(y)

        y = self.lrl(y)

        return y





class SISRDiscriminator(tf.keras.Model):



    def __init__(self, input_shape):

        super(SISRDiscriminator, self).__init__()

        self.layer_list = [

            tf.keras.layers.InputLayer(input_shape),

            ConvBlock(64, 3, 1, bn_flag=False),

            ConvBlock(64, 3, 2),

            ConvBlock(128, 3, 1),

            ConvBlock(128, 3, 2),

            ConvBlock(256, 3, 1),

            ConvBlock(256, 3, 2),

            ConvBlock(512, 3, 1),

            ConvBlock(512, 3, 2),

            tf.keras.layers.Flatten(),

            tf.keras.layers.Dense(128),

            tf.keras.layers.LeakyReLU(alpha=0.3),

            tf.keras.layers.Dense(1, activation="sigmoid")

        ]



    def call(self, x):

        for layer in self.layer_list:

            x = layer(x)



        return x

def content_loss(real_fm, fake_fm):

    loss = tf.reduce_mean((real_fm - fake_fm) ** 2)

    return loss





def adversarial_loss(fake_probs):

    labels = tf.ones_like(fake_probs)

    loss = tf.keras.losses.BinaryCrossentropy()(labels, fake_probs)

    return loss





def perceptual_loss(real_fm, fake_fm, fake_probs, weights=1e-1):

    w = tf.constant(weights, dtype=tf.float32)

    return w * content_loss(real_fm, fake_fm) + adversarial_loss(fake_probs)





def discriminator_loss(real_probs, fake_probs):

    r_labels = tf.ones_like(real_probs)

    f_labels = tf.zeros_like(fake_probs)

    r_loss = tf.keras.losses.BinaryCrossentropy()(r_labels, real_probs)

    f_loss = tf.keras.losses.BinaryCrossentropy()(f_labels, fake_probs)

    loss = r_loss + f_loss

    return loss

@tf.function

def train_step(hr_imgs,lr_imgs, G, D, E, G_opt, D_opt):



    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

        sr_imgs = G(lr_imgs)



        real_fm = E((hr_imgs + 1) / 2.0)  # VGG的输入是[0,1]

        fake_fm = E((sr_imgs + 1) / 2.0)



        real_probs = D(hr_imgs)

        fake_probs = D(sr_imgs)



        gen_loss = perceptual_loss(real_fm, fake_fm, fake_probs)

        disc_loss = discriminator_loss(real_probs, fake_probs)



    grad_G = gen_tape.gradient(gen_loss, G.trainable_variables)

    grad_D = disc_tape.gradient(disc_loss, D.trainable_variables)



    G_opt.apply_gradients(zip(grad_G, G.trainable_variables))

    D_opt.apply_gradients(zip(grad_D, D.trainable_variables))

    return gen_loss, disc_loss





def inference(G, i, target_img=target_img, output_path=INFER_OUTPUT_PATH):

    img = load_img(target_img, LR_IMAGE_SIZE)



    img = img * 2.0 - 1.0

    img = img[tf.newaxis, :]

    img = G(img, training=False)

    img = tf.squeeze(img)



    if not os.path.exists(output_path):

        os.mkdir(output_path)

    output_dir = os.path.join(output_path, str(i) + ".jpg")

    tf.keras.preprocessing.image.save_img(output_dir, img)

def G_init(G,G_opt,data_gen,n_epoch_init,batch_size=2):

    mse_list = []

    for epoch in range(n_epoch_init):

        start = time.time()

        for step, (hr_imgs,lr_imgs) in enumerate(data_gen):

            with tf.GradientTape() as tape:

                sr_imgs = G(lr_imgs)

                mse_loss = content_loss(hr_imgs,sr_imgs)

            grad = tape.gradient(mse_loss,G.trainable_variables)

            G_opt.apply_gradients(zip(grad, G.trainable_variables))

            if step % 100 == 0:

                print("epoch %d/%d, step %d, mse_loss %lf" % (epoch, n_epoch_init, step, mse_loss.numpy()))

        end = time.time()

        print("epoch %d, time %f, mse_loss %f" % (epoch, end - start, mse_loss.numpy()))

        

        mse_list.append(mse_loss.numpy())

        inference(G, epoch, output_path = INIT_INFER_OUTPUT_PATH)



    return mse_list
lr_init = 1e-4

decay_every = 2

decay_rate = 0.99



train_epoch = 30

init_epoch = 2

batch_size = 32



infer_every = 1

G = SISRGenerator(4)



lr = tf.Variable(lr_init,dtype=tf.float32)

G_init_opt = tf.keras.optimizers.Adam(lr)

data_gen = get_data_gen(PATH_PATTERN,batch_size, 1)



# mse_list = G_init(G,G_init_opt,data_gen,init_epoch,batch_size)



# generateGIF(INIT_INFER_OUTPUT_PATH, "init.gif")

# plt.plot([i for i in range(len(mse_list))],mse_list)

# plt.grid()

# plt.savefig("mse_loss.png")
E = vgg_extractor("block3_pool")

D = SISRDiscriminator((*LR_IMAGE_SIZE, 3))

G_opt = tf.keras.optimizers.Adam(lr)

D_opt = tf.keras.optimizers.Adam(lr)

# 

G_loss = []

D_loss = []

best_g_loss = 4.0



for epoch in range(train_epoch):

    start = time.time()

    for step, (hr_imgs, lr_imgs) in enumerate(data_gen):



        l = train_step(hr_imgs,lr_imgs, G, D, E, G_opt, D_opt)

        G_loss.append(l[0].numpy())

        D_loss.append(l[1].numpy())

        

        if step % 100 == 0:

            print("epoch %d, step %d, G_loss = %f D_loss=%f" %

                (epoch,step, G_loss[-1], D_loss[-1]))

            

        if step % 50 == 0 and G_loss[-1] < best_g_loss:

            tf.keras.models.save_model(G, "generator.tf")

            best_g_loss = G_loss[-1]



    end = time.time()

    print("epoch %d, time %d" % (epoch, end - start))

    if epoch != 0 and epoch % infer_every == 0:

        inference(G, epoch)



    if epoch != 0 and epoch % decay_every == 0:

        new_lr_decay = decay_rate ** (epoch / decay_every)

        lr.assign(lr_init * new_lr_decay)

        print("lr = %e" % (lr_init * new_lr_decay))

   



plot_loss(G_loss, D_loss)

generateGIF(INFER_OUTPUT_PATH, "final.gif")



plt.imshow(plt.imread("/kaggle/working/final.gif"))