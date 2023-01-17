import json

import codecs

import requests

import numpy as np

import pandas as pd 

from PIL import Image

from tqdm import tqdm

from io import BytesIO
# get links and stuff from json

jsonData = []

JSONPATH = "../input/face_detection.json"

with codecs.open(JSONPATH, 'rU', 'utf-8') as js:

    for line in js:

        jsonData.append(json.loads(line))



print(f"{len(jsonData)} image found!")



print("Sample row:")

jsonData[0]
images = []

for data in tqdm(jsonData):

    response = requests.get(data['content'])

    img = np.asarray(Image.open(BytesIO(response.content)))

    images.append([img, data["annotation"]])
np.save('images.npy', images)