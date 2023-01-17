#listing directories
import os
print(os.listdir("../input/"))
#loading our trained model
from keras.models import load_model
model_path = "../input/plant-diseases-classification-using-alexnet/AlexNetModel.hdf5"
model = load_model(model_path)
model.summary()
#creating a dictionary of classes
class_dict = {'Apple___Apple_scab': 0,
 'Apple___Black_rot': 1,
 'Apple___Cedar_apple_rust': 2,
 'Apple___healthy': 3,
 'Blueberry___healthy': 4,
 'Cherry_(including_sour)___Powdery_mildew': 5,
 'Cherry_(including_sour)___healthy': 6,
 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 7,
 'Corn_(maize)___Common_rust_': 8,
 'Corn_(maize)___Northern_Leaf_Blight': 9,
 'Corn_(maize)___healthy': 10,
 'Grape___Black_rot': 11,
 'Grape___Esca_(Black_Measles)': 12,
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 13,
 'Grape___healthy': 14,
 'Orange___Haunglongbing_(Citrus_greening)': 15,
 'Peach___Bacterial_spot': 16,
 'Peach___healthy': 17,
 'Pepper,_bell___Bacterial_spot': 18,
 'Pepper,_bell___healthy': 19,
 'Potato___Early_blight': 20,
 'Potato___Late_blight': 21,
 'Potato___healthy': 22,
 'Raspberry___healthy': 23,
 'Soybean___healthy': 24,
 'Squash___Powdery_mildew': 25,
 'Strawberry___Leaf_scorch': 26,
 'Strawberry___healthy': 27,
 'Tomato___Bacterial_spot': 28,
 'Tomato___Early_blight': 29,
 'Tomato___Late_blight': 30,
 'Tomato___Leaf_Mold': 31,
 'Tomato___Septoria_leaf_spot': 32,
 'Tomato___Spider_mites Two-spotted_spider_mite': 33,
 'Tomato___Target_Spot': 34,
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 35,
 'Tomato___Tomato_mosaic_virus': 36,
 'Tomato___healthy': 37}

class_names = list(class_dict.keys())
from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np

# dimensions of our images
img_width, img_height = 224, 224
images_dir = "../input/test-plant-diseases-data/test plant diseases data"
# predicting images
import pandas as pd
test_datagen = image.ImageDataGenerator(rescale=1./255)
img_batch = test_datagen.flow_from_directory(images_dir, target_size=(img_width, img_height), shuffle=False)
predictions = model.predict_generator(img_batch, steps=1)
filenames = img_batch.filenames
predicted_class_indices = np.argmax(predictions,axis=1)
classnames = []
for i in range(10):
    classnames.append(class_names[predicted_class_indices.item(i)])
results = pd.DataFrame({"Filename":filenames,
                      "Prediction":classnames})
results.head(10)
# predicting single image
image_path = images_dir+"/Test Plant Diseases Data/corn-blight-tr.JPG"
new_img = image.load_img(image_path, target_size=(img_width, img_height))
img = image.img_to_array(new_img)
img = np.expand_dims(img, axis=0)
img = img/255

img_class = model.predict_classes(img) #returns ndim np_array
img_class_index = img_class.item() #extracting value(s)
classname = class_names[img_class_index]

img_prob = model.predict_proba(img) #returns numpy array of class probabilities
prediction_prob = img_prob.max()

pred_dict = {"Class":classname, "Probability":prediction_prob}
print(pred_dict)

#ploting image with predicted class name        
plt.figure(figsize = (4,4))
plt.imshow(new_img)
plt.axis('off')
plt.title(classname)
plt.show()

