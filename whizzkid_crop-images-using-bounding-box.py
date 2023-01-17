import numpy as np # linear algebra

import xml.etree.ElementTree as ET # for parsing XML

import matplotlib.pyplot as plt # to show images

from PIL import Image # to read images

import os

import glob
root_images="../input/all-dogs/all-dogs/"

root_annots="../input/annotation/Annotation/"
all_images=os.listdir("../input/all-dogs/all-dogs/")

print(f"Total images : {len(all_images)}")



breeds = glob.glob('../input/annotation/Annotation/*')

annotation=[]

for b in breeds:

    annotation+=glob.glob(b+"/*")

print(f"Total annotation : {len(annotation)}")



breed_map={}

for annot in annotation:

    breed=annot.split("/")[-2]

    index=breed.split("-")[0]

    breed_map.setdefault(index,breed)

    

print(f"Total Breeds : {len(breed_map)}")
def bounding_box(image):

    bpath=root_annots+str(breed_map[image.split("_")[0]])+"/"+str(image.split(".")[0])

    tree = ET.parse(bpath)

    root = tree.getroot()

    objects = root.findall('object')

    for o in objects:

        bndbox = o.find('bndbox') # reading bound box

        xmin = int(bndbox.find('xmin').text)

        ymin = int(bndbox.find('ymin').text)

        xmax = int(bndbox.find('xmax').text)

        ymax = int(bndbox.find('ymax').text)

        

    return (xmin,ymin,xmax,ymax)
plt.figure(figsize=(10,10))

for i,image in enumerate(all_images):

    bbox=bounding_box(image)

    im=Image.open(os.path.join(root_images,image))

    im=im.crop(bbox)

    

    plt.subplot(3,3,i+1)

    plt.axis("off")

    plt.imshow(im)    

    if(i==8):

        break