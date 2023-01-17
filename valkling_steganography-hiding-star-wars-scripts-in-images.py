import numpy as np
import pandas as pd
from skimage.io import imread, imshow
from skimage.transform import rescale
import skimage
import bitarray
import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))

All_SW_Scripts = ""

def TextToString(txt):
    with open (txt, "r") as file:
        data=file.readlines()
        script = ""
        for x in data[1:]:
            x = x.replace('"','').replace("\n"," \n ").split(' ')
            x[1] += ":"
            script += " ".join(x[1:-1]).replace("\n"," \n ")
        return script
    
All_SW_Scripts += TextToString("../input/star-wars-movie-scripts/SW_EpisodeIV.txt")
All_SW_Scripts += TextToString("../input/star-wars-movie-scripts/SW_EpisodeV.txt")
All_SW_Scripts += TextToString("../input/star-wars-movie-scripts/SW_EpisodeVI.txt")

print(All_SW_Scripts[:1000])
deathstar_img = imread("../input/star-wars-steganography-images/Death-Star.jpg")

plt.figure(figsize=(10, 10))
plt.imshow(deathstar_img)

print("Image is "+str(deathstar_img.shape[0])+" by "+str(deathstar_img.shape[1])+" pixels with "+str(deathstar_img.shape[2])+" color channels")
def MessageToBits(message):
    #tag message (and pad w/ spaces till 10 characters)
    tag = "{:<10}".format(str(len(message)*8))
    message = tag+message
    #convert to bits
    code = bitarray.bitarray()
    code.frombytes(message.encode('utf-8'))
    code = "".join(['1' if x == True else '0' for x in code.tolist()])
    return code

def CheckBitSize(img, message):
    h = img.shape[0]
    w = img.shape[1]
    try:
        c = img.shape[2]
    except:
        c = 1
    image_max_size = h*w*c*2
    string_size = len(message)
    print("Message is "+str(string_size/8000)+" KB and image can fit "+str(image_max_size/8000)+" KB of data")
    if string_size > image_max_size:
        print("Message is too big to be encoded in image")
        return False
    else:
        print("Image can be encoded with message. Proceed")
        return True
    
CheckBitSize(deathstar_img, MessageToBits(All_SW_Scripts))
    
%%time
def EncodeImage(img, message):
    code = MessageToBits(message)
    if CheckBitSize(img, code):
        shape = img.shape
        img = img.flatten()
        code = list(code)
        code_len = len(code)
        for i,x in enumerate(img):
            if i*2 <code_len:
                zbits = list('{0:08b}'.format(x))[:6]+code[i*2:i*2+2]
                img[i] = int("".join(zbits), 2)
            else:
                return img.reshape(shape)
        return img.reshape(shape)

encoded_img = EncodeImage(deathstar_img, All_SW_Scripts)
def CompareTwoImages(img1,img2):
    fig=plt.figure(figsize=(20, 20))

    fig.add_subplot(2, 2, 1)
    plt.imshow(img1)
    fig.add_subplot(2, 2, 2)
    plt.imshow(img2)

    plt.show()
CompareTwoImages(deathstar_img, encoded_img)
print(deathstar_img[200][200])
print(encoded_img[200][200])
%%time
def DecodeImage(img):
    bit_message = ""
    bit_count = 0
    bit_length = 200
    for i,x in enumerate(img):
        for j,y in enumerate(x):
            for k,z in enumerate(y):
                zbits = '{0:08b}'.format(z)
                bit_message += zbits[-2:]
                bit_count += 2
                if bit_count == 80:
                    try:
                        decoded_tag = bitarray.bitarray(bit_message).tobytes().decode('utf-8')
                        bit_length = int(decoded_tag)+80
                        bit_message = ""
                    except:
                        print("Image does not have decode tag. Image is either not encoded or, at least, not encoded in a way this decoder recognizes")
                        return
                elif bit_count >= bit_length:
                    return bitarray.bitarray(bit_message).tobytes().decode('utf-8')

decoded_message = DecodeImage(encoded_img)
print(decoded_message[:1000])
print(decoded_message == All_SW_Scripts)
skimage.io.imsave("Death_Star_With_Scripts.jpg", encoded_img)
plans_img = imread("../input/star-wars-steganography-images/Deathstar_blueprint.jpg")

plt.figure(figsize=(10, 10))
plt.imshow(plans_img, cmap="gray")

print("Image is "+str(plans_img.shape[0])+" by "+str(plans_img.shape[1])+" pixels")
r2d2_img = imread("../input/star-wars-steganography-images/R2D2.jpg")

plt.figure(figsize=(10, 10))
plt.imshow(r2d2_img)

print("Image is "+str(r2d2_img.shape[0])+" by "+str(r2d2_img.shape[1])+" pixels with "+str(r2d2_img.shape[2])+" color channels")
def ImageToBits(img):
    try:
        channels = str(img.shape[2])
    except:
        channels = "1"
    tag = "{:<20}".format("img,"+str(img.shape[0])+","+str(img.shape[1])+","+channels)
    #convert tag to bits
    code = bitarray.bitarray()
    code.frombytes(tag.encode('utf-8'))
    tag = "".join(['1' if x == True else '0' for x in code.tolist()])
    # combine tag bits with the images bits
    bits_string = tag + ''.join(['{0:08b}'.format(x) for x in list(img.flatten())])
    return bits_string
    
test_image_bits = ImageToBits(r2d2_img)
print(test_image_bits[:1000])
def MessageToBits(message):
    #tag message (and pad w/ spaces till 20 characters)
    tag = "{:<20}".format("text,"+str(len(message)*8))
    message = tag+message
    #convert to bits
    code = bitarray.bitarray()
    code.frombytes(message.encode('utf-8'))
    code = "".join(['1' if x == True else '0' for x in code.tolist()])
    return code
def BitsToImage(bits_string):
    try:
        tag = bits_string[:160]
        tag = bitarray.bitarray(tag).tobytes().decode('utf-8')
        tag = tag.split(",")
        image_bits = bits_string[160:]
        h = int(tag[1])
        w = int(tag[2])
        c = int(tag[3])
        image_bits = np.asarray([int(image_bits[i:i+8], 2) for i in range(0, len(image_bits), 8)])
        if c == 1:
            image_bits = image_bits.reshape([h,w])
        else:
            image_bits = image_bits.reshape([h,w,c])
        return image_bits.astype(np.uint8)
    except:
        print('Not a string of image bits')
    
    
output_test = BitsToImage(test_image_bits)

plt.figure(figsize=(10, 10))
plt.imshow(output_test)
%%time
def EncodeImage(img, message):
    if type(message) is str:
        code = MessageToBits(message)
    else:
        code = ImageToBits(message)
    if CheckBitSize(img, code):
        shape = img.shape
        img = img.flatten()
        code = list(code)
        code_len = len(code)
        for i,x in enumerate(img):
            if i*2 <code_len:
                zbits = list('{0:08b}'.format(x))[:6]+code[i*2:i*2+2]
                img[i] = int("".join(zbits), 2)
            else:
                return img.reshape(shape)
        return img.reshape(shape)

encoded_img = EncodeImage(r2d2_img, plans_img)
CompareTwoImages(r2d2_img, encoded_img)
print(r2d2_img[200][200])
print(encoded_img[200][200])
%%time
def DecodeImage(img):
    bit_message = ""
    bit_count = 0
    bit_length = 200
    grey = len(img.shape) == 2
    for i,x in enumerate(img):
        for j,y in enumerate(x):
            if grey:
                y = [y]
            for k,z in enumerate(y):
                zbits = '{0:08b}'.format(z)
#                 print(zbits[-2:])
                bit_message += zbits[-2:]
                bit_count += 2
                if bit_count == 160:
                    try:
                        decoded_tag = bitarray.bitarray(bit_message).tobytes().decode('utf-8').split(",")
                        message_type = decoded_tag[0]
                        if message_type == "text":  
                            bit_length = int(decoded_tag[1])+160
                            bit_message = ""
                        else:
                            bit_length = (int(decoded_tag[1])*int(decoded_tag[2])*int(decoded_tag[3])*8)+160
                    except:
                        print("Image does not have decode tag. Image is either not encoded or, at least, not encoded in a way this decoder recognizes")
                        return
                elif bit_count >= bit_length:
                    if message_type == "text":
                        return bitarray.bitarray(bit_message).tobytes().decode('utf-8')
                    else:
                        return BitsToImage(bit_message)

decoded_img = DecodeImage(encoded_img)
plt.figure(figsize=(10, 10))
plt.imshow(decoded_img, cmap="gray")
New_Hope_Script = TextToString("../input/star-wars-movie-scripts/SW_EpisodeIV.txt")
death_star_hd_img = imread("../input/star-wars-steganography-images/death_star_HD.jpg")

plt.figure(figsize=(10, 10))
plt.imshow(death_star_hd_img)

print("Image is "+str(death_star_hd_img.shape[0])+" by "+str(death_star_hd_img.shape[1])+" pixels with "+str(death_star_hd_img.shape[2])+" color channels")
xwing_img = imread("../input/star-wars-steganography-images/x-wing.jpg")

plt.figure(figsize=(10, 10))
plt.imshow(xwing_img)

print("Image is "+str(xwing_img.shape[0])+" by "+str(xwing_img.shape[1])+" pixels with "+str(xwing_img.shape[2])+" color channels")
CheckBitSize(xwing_img, ImageToBits(r2d2_img))
CheckBitSize(death_star_hd_img, ImageToBits(xwing_img))
r_xwing_img = (rescale(xwing_img, 0.95) * 255).astype(np.uint8)

plt.figure(figsize=(10, 10))
plt.imshow(r_xwing_img)

print("Image is "+str(r_xwing_img.shape[0])+" by "+str(r_xwing_img.shape[1])+" pixels with "+str(r_xwing_img.shape[2])+" color channels")
CheckBitSize(death_star_hd_img, ImageToBits(r_xwing_img))
CheckBitSize(death_star_hd_img, ImageToBits(r_xwing_img))
CheckBitSize(r_xwing_img, ImageToBits(r2d2_img))
CheckBitSize(r2d2_img, ImageToBits(plans_img))
CheckBitSize(plans_img, MessageToBits(New_Hope_Script))
%%time
nested_img = EncodeImage(plans_img, New_Hope_Script)
print("1st encode done")
nested_img = EncodeImage(r2d2_img, nested_img)
print("2nd encode done")
nested_img = EncodeImage(r_xwing_img, nested_img)
print("3rd encode done")
nested_img = EncodeImage(death_star_hd_img, nested_img)
print("4th encode done")
CompareTwoImages(death_star_hd_img, nested_img)
%%time
decoded_xwing = DecodeImage(nested_img)
print("Decoded Death Star")
CompareTwoImages(r_xwing_img, decoded_xwing)
decoded_r2d2 = DecodeImage(decoded_xwing)
print("Decoded X-Wing")
CompareTwoImages(r2d2_img, decoded_r2d2)
decoded_plans = DecodeImage(decoded_r2d2)
print("Decoded R2D2")
CompareTwoImages(plans_img, decoded_plans)
decoded_new_hope = DecodeImage(decoded_plans)
print("Decoded Death Star Plans")
print(decoded_new_hope[:1000])
skimage.io.imsave("Encoded_Death_Star_HD.jpg", nested_img)
skimage.io.imsave("Encoded_X-Wing.jpg", decoded_xwing)
skimage.io.imsave("Encoded_R2D2.jpg", decoded_r2d2)
skimage.io.imsave("Encoded_Death_Star_Plans.jpg", decoded_plans)
%%time
import random
def ScrambleEncodedImage(img):
    shape = img.shape
    img = img.flatten()
    for i,x in enumerate(img):
        r = [str(random.randint(0,1)),str(random.randint(0,1))]
        zbits = list('{0:08b}'.format(x))[:6]+r
        img[i] = int("".join(zbits), 2)
    return img.reshape(shape)

scrambled_img = ScrambleEncodedImage(decoded_r2d2)
CompareTwoImages(decoded_r2d2, scrambled_img)