from IPython.display import Image

Image("../input/animages/1.png")
# importing libraries



import numpy as np

import matplotlib.pyplot as plt

import os



# Libraries for TensorFlow

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.preprocessing import image

from tensorflow.keras import models, layers



# Library for Transfer Learning

from tensorflow.keras.applications import resnet50



print("Importing libraries completed.")
# listing the folders containing images



# Main Dataset

main_folder="../input/animal10/Animals-10"

class_names=os.listdir(main_folder)

print(class_names)



# Validation Dataset

validation_folder="../input/validationimage"

val_class_names=os.listdir(validation_folder)

print(val_class_names)
# Preparing Training image data (image array and class name) for processing



# Declaring variables

x=[] # to store array value of the images

y=[] # to store the labels of the images



for folder in os.listdir(main_folder):

    image_list=os.listdir(main_folder+"/"+folder)

    for img_name in image_list:

        # Loading images

        img=image.load_img(main_folder+"/"+folder+"/"+img_name,target_size=(100,100))

        

        # Converting to arrary

        img=image.img_to_array(img)

        

        # Transfer Learning: this is to apply preprocess of resnet to our images before passing it to resnet

        img=resnet50.preprocess_input(img) #  Optional step

        

        # Appending the arrarys

        x.append(img) # appending image array

        y.append(class_names.index(folder)) # appending class index to the array

        

print("Preparing Training Dataset Completed.")
# Preparing validation images data (image array and class name) for processing



# Declaring variables

val_images=[]

val_image_label=[] # to store the labels of the images



for folder in os.listdir(validation_folder):

    image_list=os.listdir(validation_folder+"/"+folder)

    for img_name in image_list:

        # Loading images

        img=image.load_img(validation_folder+"/"+folder+"/"+img_name,target_size=(100,100))

        

        # Converting to arrarys

        img=image.img_to_array(img)



        # Transfer Learning: this is to apply preprocess of resnet to our images before passing it to resnet

        img=resnet50.preprocess_input(img) #  Optional step

        

        # Appending arrays

        val_images.append(img) # appending image array

        val_image_label.append(val_class_names.index(folder))

        

print("Preparing Validation Dataset Completed.")
# Verifying the output



# Training Dataset

print("Training Dataset")



x=np.array(x) # Converting to np arrary to pass to the model

print(x.shape)



y=to_categorical(y) # onehot encoding of the labels

# print(y)

print(y.shape)





# Validation Dataset

print("Validation Dataset")



val_images=np.array(val_images) 

print(val_images.shape)



val_image_label=to_categorical(val_image_label) # onehot encoding of the labels)

print(val_image_label.shape)
# Check properties of the model that we are going to use for Transfer Learning



print("Summary of default ResNet50 model.\n")



# we are using resnet50 for transfer learnin here. So we have imported it

from tensorflow.keras.applications import resnet50



# initializing model with weights='imagenet'i.e. we are carring its original weights

model_resnet=resnet50.ResNet50(weights='imagenet')



# display the summary to see the properties of the model

model_resnet.summary()
# Modelling WITH Transfer Learning



# Here we will prepare model as per our requirements



print("Summary of Custom ResNet50 model.\n")

print("1) We setup input layer and 2) We removed top (last) layer. \n")



# let us prepare our input_layer to pass our image size. default is (224,224,3). we will change it to (100,100,3)

input_layer=layers.Input(shape=(100,100,3))



# initialize the transfer model ResNet50 with appropriate properties per our need.

# we are passing paramers as following

# 1) weights='imagenet' - Using this we are carring weights as of original weights.

# 2) input_tensor to pass the ResNet50 using input_tensor

# 3) we want to change the last layer so we are not including top layer

resnet_model=resnet50.ResNet50(weights='imagenet',input_tensor=input_layer,include_top=False)



# See the summary of the model with our properties.

resnet_model.summary()
# access the current last layer of the model and add flatten and dense after it



print("Summary of Custom ResNet50 model.\n")

print("1) We flatten the last layer and added 1 Dense layer and 1 output layer.\n")



last_layer=resnet_model.output # we are taking last layer of the model



# Add flatten layer: we are extending Neural Network by adding flattn layer

flatten=layers.Flatten()(last_layer) 



# Add dense layer

# dense1=layers.Dense(100,activation='relu')(flatten)



# Add dense layer to the final output layer

output_layer=layers.Dense(10,activation='softmax')(flatten)



# Creating modle with input and output layer

model=models.Model(inputs=input_layer,outputs=output_layer)



# Summarize the model

model.summary()
# we will freez all the layers except the last layer



# we are making all the layers intrainable except the last layer

print("We are making all the layers intrainable except the last layer. \n")

for layer in model.layers[:-1]:

    layer.trainable=False

model.summary()
# Train the Model



from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2,random_state=5)

# print(xtrain)

# print(xtest)

# print(ytrain)

# print(ytest)



print("Splitting data for train and test completed.")
# Compiling Model



model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])



print("Model compilation completed.")
# Fit the Model



model.fit(xtrain,ytrain,epochs=20,batch_size=64,verbose=True,validation_data=(xtest,ytest))



print("Fitting the model completed.")
# This function helps to predict individual image supplied to it

def predict(img_name):

    img=image.load_img(img_name,target_size=(100,100))

    img=image.img_to_array(img)

    plt.imshow(img.astype('int32'))

    plt.show()

    img=resnet50.preprocess_input(img)

    prediction=model.predict(img.reshape(1,100,100,3))

    output=np.argmax(prediction)

    print(class_names[output])



# This function plots the image supplied in array

def plot_image(i, predictions_array, true_label, img): # taking index and 3 arrays viz. prediction array, true label array and image array

    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]

    plt.grid(False)

    plt.xticks([])

    plt.yticks([])

    

#     plt.imshow(img) # showing b/w image

    plt.imshow(img.astype('int32'))

    

#     print(*predictions_array)

    predicted_label=np.argmax(predictions_array)

    true_label=np.argmax(true_label)



#     print(predicted_label)

#     print(true_label)



    if predicted_label == true_label: #setting up label color

        color='green' # correct then blue colour

    else:

        color='red' # wrong then red colour

    plt.xlabel("{} {:2.0f}% ({})".format(val_class_names[predicted_label], 100*np.max(predictions_array), val_class_names[true_label]), color=color)



# This function plots bar chart supplied in the array data

def plot_value_array(i, predictions_array, true_label): # taking index along with predictions and true label array

    predictions_array, true_label = predictions_array[i], true_label[i]

    plt.grid(False)

    plt.xticks([])

    plt.yticks([])

    predicted_label=np.argmax(predictions_array)

    true_label=np.argmax(true_label)

#     print(predicted_label)

#     print(true_label)

    if predicted_label == 0:

        predicted_label=1

    if true_label == 0:

        true_label=1

    

    thisplot=plt.bar(range(10), predicted_label, color='seashell')

    plt.ylim([0,1])



    thisplot[predicted_label].set_color('red')

    thisplot[true_label].set_color('green')
# Preparing prediction arrary

predictions=[]



for img in val_images:

    img=img.reshape(1,100,100,3)

    predictions.append(model.predict(img))
# Prediction of individual images taken from internet



# Butterfly

!wget -q https://www.futurity.org/wp/wp-content/uploads/2019/02/viceroy-butterfly_1600.jpg

predict('viceroy-butterfly_1600.jpg') 

    

# Cat

!wget -q https://static.scientificamerican.com/sciam/cache/file/92E141F8-36E4-4331-BB2EE42AC8674DD3_source.jpg

predict('92E141F8-36E4-4331-BB2EE42AC8674DD3_source.jpg')



# call the function



# defining parameters to pass to function

i=22 # image number 12. You may change value of i for play around

plt.figure(figsize=(6,3))

plt.subplot(1,2,1)



plot_image(i,predictions, val_image_label, val_images)

plt.subplot(1,2,2)

plot_value_array(i, predictions, val_image_label)

plt.show()
# Prediction on Validation images



# Declaring variables

num_rows=6

num_cols=4

num_images=num_rows*num_cols



plt.figure(figsize=(2*2*num_cols,2*num_rows))

for i in range(num_images):

    plt.subplot(num_rows, 2*num_cols, 2*i+1)



    plot_image(i,predictions, val_image_label, val_images)

    

    plt.subplot(num_rows, 2*num_cols, 2*i+2)



    plot_value_array(i, predictions, val_image_label)

plt.show()
print("Notebook completed.")