import tensorflow as tf
import os
import glob
import time
import random
from matplotlib import pyplot as plt
#from kaggle_datasets import KaggleDatasets
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
#gcs_path = KaggleDatasets().get_gcs_path('Anime Sketch Colorization Pair') 
img_path = glob.glob('../input/anime-sketch-colorization-pair/data/train/*.png')
#len(img_path)
#gcs_path
Batch_Number = 8
EPOCHS = 150
def read_img(img_path):
    img_value = tf.io.read_file(img_path)
    img = tf.image.decode_png(img_value,channels=3)
    return img

def load_img(img_path):
    imgs = read_img(img_path)
    w = tf.shape(imgs)[1]
    w = w//2
    input_image = imgs[:,:w,:]
    musk_image = imgs[:,w:,:]
    input_image  = tf.image.resize(input_image,(512,512))
    musk_image = tf.image.resize(musk_image,(512,512))
    if tf.random.uniform(())>0.5:
        input_image = tf.image.flip_left_right(input_image)
        musk_image = tf.image.flip_left_right(musk_image)
    input_image,musk_image = normalize(input_image,musk_image)
    return musk_image,input_image
#归一化，将所有值归一到[-1，1]
def normalize(input_image,musk_image):
    input_image = tf.cast(input_image,tf.float32) / 127.5 - 1 
    musk_image = tf.cast(musk_image,tf.float32) / 127.5 - 1
    return input_image,musk_image
dataset = tf.data.Dataset.from_tensor_slices(img_path)
train = dataset.map(load_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)

BATCH_SIZE = 16
BUFFER_SIZE = 200
train_dataset = train.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
train_dataset = train_dataset.prefetch(buffer_size = tf.data.experimental.AUTOTUNE)
plt.figure(figsize=(8,5))
for img,musk in train_dataset.take(1):
    plt.subplot(1,2,1)
    plt.imshow(tf.keras.preprocessing.image.array_to_img(img[0]))
    plt.subplot(1,2,2)
    plt.imshow(tf.keras.preprocessing.image.array_to_img(musk[0]))
img_path_test = glob.glob('../input/anime-sketch-colorization-pair/data/val/*.png')
dataset_test = tf.data.Dataset.from_tensor_slices(img_path_test)

def load_img_test(img_path):
    imgs = read_img(img_path)
    w = tf.shape(imgs)[1]
    w = w//2
    input_image = imgs[:,:w,:]
    musk_image = imgs[:,w:,:]
    input_image  = tf.image.resize(input_image,(256,256))
    musk_image = tf.image.resize(musk_image,(256,256))
    input_image,musk_image = normalize(input_image,musk_image)
    return musk_image,input_image
test = dataset_test.map(load_img_test)
dataset_test = test.batch(BATCH_SIZE)
plt.figure(figsize=(8,5))
for img,musk in dataset_test.take(1):
    plt.subplot(1,2,1)
    plt.imshow(tf.keras.preprocessing.image.array_to_img(img[0]))
    plt.subplot(1,2,2)
    plt.imshow(tf.keras.preprocessing.image.array_to_img(musk[0]))
def down(filters,size,apply_bn=True):
    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.Conv2D(filters,size,strides=2,padding='same',use_bias='False')
    )
    if apply_bn:
        model.add(
        tf.keras.layers.BatchNormalization()
        )
    model.add(tf.keras.layers.LeakyReLU())
    return model

def up_sample(filters,size,apply_dropout=False):
    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.Conv2DTranspose(filters,size,strides=2,padding='same',use_bias=False)
    )
    if apply_dropout:
        model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.ReLU())
    return model
def Generator():
    inputs =tf.keras.layers.Input(shape=[256,256,3])
    down_stack=[
        down(64,4,apply_bn=False),      #128*128*64
        down(128,4),     #64*64*128
        down(256,4),     #32*32*256
        down(512,4),     #16*16*512
        down(512,4),     #8*8*512
        down(512,4),     #4*4*512
        down(512,4),     #2*2*512
        down(512,4),     #1*1*512
    ]
    
    up_stack=[
        up_sample(512,4,apply_dropout=True),   #2*2*512
        up_sample(512,4,apply_dropout=True),   #4*4*512
        up_sample(512,4,apply_dropout=True),   #8*8*512
        up_sample(512,4),   #16*16*512
        up_sample(256,4),   #32*32*256
        up_sample(128,4),   #64*64*128
        up_sample(64,4),   #128*128*64
    ]
    
    last = tf.keras.layers.Conv2DTranspose(3,4,strides=2,padding='same',activation = 'tanh')
    
    x = inputs
    skips = []
    for d in down_stack:
        x = d(x)
        skips.append(x)
        
    skips = reversed(skips[:-1])
    for up,skip in zip(up_stack,skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x,skip])
        
    x = last(x)
    return tf.keras.Model(inputs = inputs,outputs = x)
generator = Generator()
#tf.keras.utils.plot_model(generator,show_shapes=True)
LAMBDA = 10
def Disc():
    inp = tf.keras.layers.Input(shape = [256,256,3],name = 'input_image')
    target = tf.keras.layers.Input(shape = [256,256,3],name = 'target_image')
    x = tf.keras.layers.Concatenate()([inp,target])
    
    down1 = down(64,4,False)(x)  #128*128*64
    down2 = down(128,4)(down1)  #64*64*128
    down3 = down(256,4)(down2)  #32*32*256
    
    conv = tf.keras.layers.Conv2D(512,4,strides = 1,padding = 'same',use_bias = False)(down3)
    batchnor1 = tf.keras.layers.BatchNormalization()(conv)
    leaky_relu = tf.keras.layers.LeakyReLU()(batchnor1)
    last = tf.keras.layers.Conv2D(1,4,strides = 1,padding = 'same')(leaky_relu)
    return tf.keras.Model(inputs = [inp,target], outputs = last)
discriminator = Disc()
#tf.keras.utils.plot_model(discriminator,show_shapes=True)
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
def generate_loss(disc_generate_output,gen_output,target):
    gen_loss = loss_object(tf.ones_like(disc_generate_output),disc_generate_output)
    
    l1_loss = tf.reduce_mean(tf.abs(gen_output - target))
    
    total_gen_loss = gen_loss + LAMBDA * l1_loss
    return total_gen_loss, gen_loss, l1_loss
def discriminator_loss(disc_real_output,disc_generate_out):
    real_loss = loss_object(tf.ones_like(disc_real_output),disc_real_output)
    generate_loss = loss_object(tf.zeros_like(disc_generate_out),disc_generate_out)
    return real_loss + generate_loss
generator_optimizer = tf.keras.optimizers.Adam(2e-4,beta_1 = 0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4,beta_1 = 0.5)
def generate_image(model,test_input,tar):
    prediction = model(test_input,training = True)
    plt.figure(figsize=(15,15))
    display_list = [test_input[0],tar[0],prediction[0]]
    for i in range(3):
        plt.subplot(1,3,i+1)
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()
    

for example_input,example_target in dataset_test.take(1):
    generate_image(generator,example_input,example_target)
@tf.function
def train_step(input_image,target,epoch):
    with tf.GradientTape() as gen_tape,tf.GradientTape() as disc_tape:
        gen_output = generator(input_image,training = True)
        disc_real_output = discriminator([input_image,target],training = True)
        disc_generate_output = discriminator([input_image,gen_output])
        gen_total_loss ,gen_gen_loss ,gen_l1_loss = generate_loss(disc_generate_output,gen_output,target)
        disc_loss = discriminator_loss(disc_real_output,disc_generate_output)
        
        generator_gradients = gen_tape.gradient(gen_total_loss,generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss,discriminator.trainable_variables)
        generator_optimizer.apply_gradients(zip(generator_gradients,generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(discriminator_gradients,discriminator.trainable_variables))
    return gen_total_loss,disc_loss,gen_l1_loss
        
@tf.function
def train_step_tpu(input_image,target,epoch):
    with strategy.scope():
        with tf.GradientTape() as gen_tape,tf.GradientTape() as disc_tape:
            gen_output = generator(input_image,training = True)
            disc_real_output = discriminator([input_image,target],training = True)
            disc_generate_output = discriminator([input_image,gen_output])
            gen_total_loss ,gen_gen_loss ,gen_l1_loss = generate_loss(disc_generate_output,gen_output,target)
            disc_loss = discriminator_loss(disc_real_output,disc_generate_output)
        
            generator_gradients = gen_tape.gradient(gen_total_loss,generator.trainable_variables)
            discriminator_gradients = disc_tape.gradient(disc_loss,discriminator.trainable_variables)
            generator_optimizer.apply_gradients(zip(generator_gradients,generator.trainable_variables))
            discriminator_optimizer.apply_gradients(zip(discriminator_gradients,discriminator.trainable_variables))
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
def fit(train_ds,epochs,test_ds):
    for epoch in range(epochs+1):
        if epoch%2 == 0:
            checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                             discriminator_optimizer=discriminator_optimizer,
                                             generator=generator,
                                             discriminator=discriminator
                                            )
            checkpoint.save(checkpoint_prefix)
        print("Epoch: ",epoch)
        for example_input,example_target in dataset_test.take(1):
                generate_image(generator,example_input,example_target)
        for n,(input_image,target) in train_ds.enumerate():
            print('.',end ='')
            gen_total_loss, disc_loss,gen_l1_loss = train_step(input_image,target,epoch)
            print(gen_total_loss.numpy())
        print()
    

fit(train_dataset,EPOCHS,dataset_test)
dataset = tf.data.Dataset.range(20)
dataset = dataset.batch(5)
for i in dataset.enumerate():
    print(i)
