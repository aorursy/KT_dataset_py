!pip install mtcnn                               
from keras.models import load_model

from PIL import Image

import matplotlib.pyplot as plt

from mtcnn.mtcnn import MTCNN

from numpy import * 

import pandas as pd 
model = load_model('../input/facenet-keras/facenet_keras.h5')      



print(model.inputs)                                                

print(model.outputs)
def get_embedding(model, face_pixels):               

    # scale pixel values

    face_pixels = face_pixels.astype('float32')

    # standardize pixel values across channels (global)

    mean, std = face_pixels.mean(), face_pixels.std()

    face_pixels = (face_pixels - mean) / std

    # transform face into one sample

    samples = expand_dims(face_pixels, axis=0)

    # make prediction to get embedding

    yhat = model.predict(samples)

    return yhat[0]
def extract_face(filename, required_size=(160, 160)): 

    # load image from file

    image = Image.open(filename)

    # convert to RGB, if needed

    image = image.convert('RGB')

    # convert to array

    pixels = asarray(image)

    # create the detector, using default weights

    detector = MTCNN()

    # detect faces in the image

    results = detector.detect_faces(pixels)

    # extract the bounding box from the first face

    x1, y1, width, height = results[0]['box']

    # bug fix

    x1, y1 = abs(x1), abs(y1)

    x2, y2 = x1 + width, y1 + height

    # extract the face

    face = pixels[y1:y2, x1:x2]

    # resize pixels to the model size

    image = Image.fromarray(face)

    image = image.resize(required_size)

    face_array = asarray(image)

    return face_array
originalface = extract_face('../input/original-image/billgates2.jpg')

testface     = extract_face('../input/sample-image/billgates1.jpeg')
plt.imshow(originalface)   #cropped face of original image
plt.imshow(testface)  #cropped face of test image
originalembedding = get_embedding(model,originalface)    

testembedding = get_embedding(model,testface)
dist = linalg.norm(testembedding-originalembedding)    
print(dist)