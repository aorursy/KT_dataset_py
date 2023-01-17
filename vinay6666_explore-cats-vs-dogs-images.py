#getting training and testing file's directories

train_dir = '../input/train'

test_dir = '../input/test'
import os
#print cureent working directory

print(os.getcwd())
#list files and folders in train_dir directory

#print(os.listdir(train_dir))



#this will print all the image file names in this directory. So commented. 
#list files in test directory

#print(os.listdir(test_dir))
import random
image_filenames = os.listdir(train_dir)

image_sample = random.sample(image_filenames, 5)

print(image_sample)


from IPython.display import display, Image
for item in image_sample:

    image_path = os.path.join(train_dir, item)

    image = Image(filename=image_path)

    print(image_path)

    display(image)
image_filenames = os.listdir(test_dir)

image_sample = random.sample(image_filenames, 5)

print(image_sample)
for item in image_sample:

    image_path = os.path.join(test_dir, item)

    image = Image(filename=image_path)

    print(image_path)

    display(image)