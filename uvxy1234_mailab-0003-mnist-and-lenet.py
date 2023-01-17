### mAiLab_0003：MNIST and LeNet 



import numpy as np

import requests

import os

#from tqdm import tqdm_notebook as tqdm

from tqdm import tqdm as tqdm

import gzip
# Return target file size in byte (int)

def check_size(url):

    r = requests.get(url, stream=True)

    return int(r.headers['Content-Length'])



# Define helper function for download (int)

def download_file(url, filename, bar=True):

    """

    Helper method handling downloading large files 

    from `url` to `filename`. Returns a pointer to `filename`.

    """

    try:

        chunkSize = 1024

        r = requests.get(url, stream=True)

        with open(filename, 'wb') as f:

            if bar:

                pbar = tqdm(unit="B", total=check_size(url))

            for chunk in r.iter_content(chunk_size=chunkSize): 

                if chunk: # filter out keep-alive new chunks

                    if bar: 

                        pbar.update(len(chunk))

                    f.write(chunk)

        return

    except Exception as e:

        print(e)

        return
#HW0003-1: 下載以下四個檔案, 將其用解壓縮軟體解壓縮之後得到四個檔案



filename_list = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz',

                't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']

#Download

for filename in filename_list:

    download_file('http://yann.lecun.com/exdb/mnist/' + filename, filename)



print(os.listdir())
#Load image and label from xxx.gz

def read_mnist(images, labels):

    with gzip.open(labels, 'rb') as labelsFile:

        #the lable byte begin form the 0008 byte, so we set the offset to 8 for ignoring 0~7

        labels = np.frombuffer(labelsFile.read(), dtype=np.uint8, offset=8)



    with gzip.open(images,'rb') as imagesFile:

        data_size = len(labels)

        # Load flat 28x28 px images

        # the lable byte begin form the 0016 byte, so we set the offset to 8 for ignoring 0~15

        features = np.frombuffer(imagesFile.read(), dtype=np.uint8, offset=16).reshape(data_size, 28, 28)

        

    return features, labels



#print out image by a 28*28 matrix

def print_image(file):

    for i in file:

        for j in i:

            # {:02X} output the pixel numbers by two digits hexadecimal

            # example: 255 -> FF ; 14 -> 1E

            print("{:02X}".format(j), end=' ') 

        print()

    print()





#HW0003-2:  輸出 train-images.idx3-ubyte 檔案中的第一個圖，大小為 28x28

image, label = read_mnist(images=filename_list[0], labels=filename_list[1])

print('first label:{}'.format(label[0]))

print('first image:')

print_image(image[0])
#HW0003-3:  輸出 train-images.idx3-ubyte 檔案中前十個圖的平均圖，採無條件捨去，大小為 28x28。

image_avg = np.zeros((28,28))

for i in range(10):

    image_avg += image[i]



image_avg = (image_avg/10).astype('uint8')



print('average of first 10 images:')

print_image(image_avg)
#HW0003-4:  輸出 train-labels.idx1-ubyte 檔案中前十個 labels 的平均，精確度取至小數點以下兩位，採無條件捨去。

label_avg = label[0:10].mean()

print("{:.2f}".format(label_avg))
#HW0003-5:  輸出 train-images.idx3-ubyte 檔案中的第一個圖，大小為 32x32。原圖置中，多出來的地方補0。

#           (zero padding)



#def zero_padding(img):

#    pass



padded_image = np.pad(image[0], [2, 2], mode='edge')

print("Padded image")

print_image(padded_image)

#HW0003-6:  將第一張圖檔存成BMP格式

import scipy.misc

scipy.misc.imsave('first_training_image.bmp', image[0])