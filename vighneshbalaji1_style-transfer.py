import time
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from ipywidgets import IntProgress
from IPython.display import display
import h5py

import tensorflow as tf
from tensorflow.python.keras import models 
from tensorflow.python.keras import losses
from tensorflow.python.keras import layers
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.preprocessing import image as kp_image

import tensorflow.contrib.eager as tfe
tf.enable_eager_execution()
# print("Eager execution: {}".format(tf.executing_eagerly()))
print("created by: Raj Gupta")
def load_img(filename, max_size=None):
    
    image = Image.open(filename)
    
    if max_size is not None:
        factor = max_size / np.max(image.size)
        size = np.array(image.size) * factor
        size = size.astype(int)
        image = image.resize(size, Image.LANCZOS)

    return np.expand_dims(np.float32(image), axis=0)

def show_art(content_image, style_image, mixed_image):
    fig, axes = plt.subplots(1, 3, figsize=(20, 20))

    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    smooth = True

    if smooth:
        interpolation = 'sinc'
    else:
        interpolation = 'nearest'

    ax = axes.flat[0]
    ax.imshow(content_image / 255.0, interpolation=interpolation)
    ax.set_xlabel("Content")

    ax = axes.flat[1]
    ax.imshow(style_image / 255.0, interpolation=interpolation)
    ax.set_xlabel("Style")

    ax = axes.flat[2]
    ax.imshow(mixed_image / 255.0, interpolation=interpolation)
    ax.set_xlabel("Art")

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.show()

def save_image(image, filename):
  
    image = np.clip(image, 0.0, 255.0)

    image = image.astype(np.uint8)
    
    with open(filename, 'wb') as file:
        Image.fromarray(image).save(file, 'jpeg')

def load_and_process(path_to_img):
    img = load_img(path_to_img)
    img = tf.keras.applications.vgg19.preprocess_input(img)
    return img

def deprocess_img(processed_img):
    x = processed_img.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x, 0)
    assert len(x.shape) == 3, ("Invalid dimension")
    if len(x.shape) != 3:
        raise ValueError("Invalid input")
 
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]

    x = np.clip(x, 0, 255).astype('uint8')
    return x

content_layers = ['block5_conv2'] 

style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1']

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

def get_model():
  
    vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
  
    style_outputs = [vgg.get_layer(name).output for name in style_layers]
    content_outputs = [vgg.get_layer(name).output for name in content_layers]
    model_outputs = style_outputs + content_outputs
    
    return models.Model(vgg.input, model_outputs)

def calc_content_loss(base_content, target):
    return tf.reduce_mean(tf.square(base_content - target))

def gram_matrix(tensor):
    shape = tensor.shape[-1]
    #print(shape)
    num_channels = int(shape)
    #print(num_channels)
    matrix = tf.reshape(tensor, shape=[-1, num_channels])
    #print(matrix)
    gram = tf.matmul(tf.transpose(matrix), matrix)

    return gram / tf.cast(matrix.shape[0], tf.float32)     

def calc_style_loss(base_style, gram_target):
    return tf.reduce_mean(tf.square(gram_matrix(base_style) - gram_target))

def calc_loss(model, loss_weights, init_image, gram_style_features, c_features):
  
    style_weight, content_weight = loss_weights
  

    model_outputs = model(init_image)
  
    style_output_features = model_outputs[:num_style_layers]
    content_output_features = model_outputs[num_style_layers:]
  
    style_score = 0
    content_score = 0

  
    weight_per_style_layer = 1.0 / float(num_style_layers)
    for target_style, comb_style in zip(gram_style_features, style_output_features):
        style_score += weight_per_style_layer * calc_style_loss(comb_style[0], target_style)
    
  
    weight_per_content_layer = 1.0 / float(num_content_layers)
    for target_content, comb_content in zip(c_features, content_output_features):
        content_score += weight_per_content_layer* calc_content_loss(comb_content[0], target_content)
        

  
    style_score *= style_weight
    content_score *= content_weight

  
    loss = style_score + content_score
    return loss, style_score, content_score

def calc_grads(config):
    with tf.GradientTape() as gt: 
        all_loss = calc_loss(**config)
    total_loss = all_loss[0]
    return gt.gradient(total_loss, config['init_image']), all_loss

def get_features(model, c_path, s_path):
  
    c_image = load_and_process(c_path)
    s_image = load_and_process(s_path)
  
    s_outputs = model(s_image)
    c_outputs = model(c_image)
    
    s_features = [style_layer[0] for style_layer in s_outputs[:num_style_layers]]
    c_features = [content_layer[0] for content_layer in c_outputs[num_style_layers:]]
    return s_features, c_features

import IPython.display

def generate_art(c_path,s_path,epochs=100,c_weight=1e3,s_weight=1e-2): 
    
    model = get_model() 
    for layer in model.layers:
        layer.trainable = False
  
    s_features, c_features = get_features(model, c_path, s_path)
    gram_style_features = [gram_matrix(style_feature) for style_feature in s_features]
  
    init_image = load_and_process(c_path)
    init_image = tfe.Variable(init_image, dtype=tf.float32)
    opt = tf.train.AdamOptimizer(learning_rate=5, beta1=0.99, epsilon=1e-1)
    optimized_loss, art = float('inf'), None
  
    loss_weights = (s_weight, c_weight)
    config = {
           'model': model,'loss_weights': loss_weights,'init_image': init_image,
           'gram_style_features': gram_style_features,'c_features': c_features
          }
    
    norm_means = np.array([103.939, 116.779, 123.68])
    min_vals = -norm_means
    max_vals = 255 - norm_means 
    
    ctr = 1
    progress = IntProgress(min=0, max=epochs)
    display(progress)
    
    start_time = time.time()
    
    
    for i in range(epochs):
        grads, all_loss = calc_grads(config)
        loss, style_score, content_score = all_loss
        opt.apply_gradients([(grads, init_image)])
        resized = tf.clip_by_value(init_image, min_vals, max_vals)
        init_image.assign(resized)
        progress.value += 1  
        time.sleep(.1)
        art = init_image.numpy()
        art = deprocess_img(art)
        IPython.display.clear_output(wait=True)
        
        print(c_path,"+",s_path)
        display(progress)
        print("......................................",((i+1)*(100/epochs)),"%")
        IPython.display.display_png(Image.fromarray(art))
        print('Iteration: {}'.format(i+1))        
        print('Combined loss: {:.4e}, ''style loss: {:.4e}, ''content loss: {:.4e}, '.format(loss, style_score, content_score))
        end_time = time.time()
        total_time = end_time - start_time
        print('Total time: {:.4f}s'.format(total_time))

    return art 

def generate(c_path):
    style = "../images/style.jpg"
    c_path = "../images/tajmahal.jpg"
    art_list = []
    for i in range(1):
        path = style
        art = generate_art(c_path,s_path, epochs=1)
        art_list.append(art)
    print("..................................................................................................")
    s_path = style
    sty = plt.imread(s_path)
    content = plt.imread(c_path)
    show_art(content, sty, art_list[i])
    save_name = "arts/"+c_path+"_"+s_path
    save_image(art_list[i], save_name)

