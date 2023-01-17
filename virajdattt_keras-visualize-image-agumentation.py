import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

import tensorflow as tf #Will be using tensor flow keras
import matplotlib.pyplot as plt # For vizualization

        
train_path = "/kaggle/input/he-dance-forms/dataset/train" #path to the images 
data = pd.read_csv("/kaggle/input/he-dance-forms/dataset/train.csv") #dataframe containg the image and class mapping
Image_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,     
                                                            horizontal_flip=True,
                                                           )

#More tranformations are defined here 
#https://keras.io/api/preprocessing/image/#imagedatagenerator-class
train_gen = Image_gen.flow_from_dataframe(dataframe=  data,                #Name of the dataframe
                                                    directory= train_path, #File location of the images 
                                                    x_col='Image',         #Column with the image name, with the extension 
                                                    y_col='target',        #Column with the name of labels
                                                    seed=40,               #Seed for reproducblity
                                                    target_size=(150, 150),#Size of all the target images
                                                    batch_size=16,          #number of batches u want to generate   
                                                    shuffle=True,
                                                    class_mode='categorical',
                                                    #save_to_dir="./" Uncomment if u want to save a copy of agumented images to disk as a png 
                                      )
X_batch_history, Y_batch_History = [], [] # Empty lists to save the previousily generated images, can also be saved to disk 

def reverse_dict(orig_dict):
    '''
        A function to reverse the keys and values in python dictonary
        
        
        [param]orgi_dict: It accepts a dictionary mapping of classes_name  to numericals 
        
        example run : - reverse_dict(train_gen.class_indices)
    '''
    
    

    new_dict = {}
    for i, j in orig_dict.items():
        #print(i, j)
        new_dict[j]=i
    return new_dict
    

def keras_batch_viz(gen_object):
    '''
    Generates a plot of 16 images 
    
    [param]gen_pbject: A  DataFrameIterator i.e ImageDataGenerator which has called it's flow_from_dataframe method
    example run :- keras_batch_viz(train_gen_test)
    '''
    
    if (gen_object.batch_size < 16):
        print("ERROR, the batch_size is less than 16 in your flow_from_dataframe method")
        return -1
    
    rdict = reverse_dict(gen_object.class_indices)
    keep_going_on = True
    while keep_going_on:
        x_batch, y_batch = next(gen_object)
    #Comment these if you don't need to save history of the vizualization 
    
        X_batch_history.append(x_batch)
        Y_batch_History.append(y_batch)
    
        fig = plt.figure(figsize=(13, 13))
        columns = 4
        rows = 4
    
        ax = []
        for i in range(1, 17): #this will loop over and display the 16 images, change according to ur needs
            image = x_batch[i-1]
            ax.append(fig.add_subplot(rows, columns, i))
            ax[-1].set_title(rdict[np.argmax(y_batch[i-1])])
            plt.imshow(image)
        plt.grid(None)
        plt.tight_layout()
        plt.show()
        
        print("Press Y/y if u want to keep going on generating batches to vizualize", end=':')
        keep_going_on = input()
        if keep_going_on.lower() != 'y':
            keep_going_on = False
            print("----- Terminating ------")
            return 0
        print("WAIT GENERATING THE NEXT BATCH", "---"*100)
    
keras_batch_viz(train_gen)

 