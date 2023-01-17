import PIL
print('Pillow Version:', PIL.__version__)
# load and show an image with Pillow
from PIL import Image
# load the image
image = Image.open('../input/johar-pic/79669868_10158168171086929_2402259235392978944_n.jpg')
# summarize some details about the image
print(image.format)
print(image.mode)
print(image.size)
# show the image
image.show()
# load and display an image with Matplotlib
from matplotlib import image
from matplotlib import pyplot
# load image as pixel array
data = image.imread('../input/johar-pic/79669868_10158168171086929_2402259235392978944_n.jpg')
# summarize shape of the pixel array
print(data.dtype)
print(data.shape)
# display the array of pixels as an image
pyplot.imshow(data)
pyplot.show()
# load image and convert to and from NumPy array
from PIL import Image
from numpy import asarray
# load the image
image = Image.open('../input/johar-pic/79669868_10158168171086929_2402259235392978944_n.jpg')
# convert image to numpy array
data = asarray(image)
# summarize shape
print(data.shape)
# create Pillow image
image2 = Image.fromarray(data)
# summarize image details
print(image2.format)
print(image2.mode)
print(image2.size)
image2
# example of saving an image in another format
from PIL import Image
# load the image
image = Image.open('../input/johar-pic/79669868_10158168171086929_2402259235392978944_n.jpg')
# save as PNG format
image.save('johar.png', format='PNG')
# load the image again and inspect the format
image2 = Image.open('johar.png')
print(image2.format)
# resize image and force a new shape
from PIL import Image
# load the image
image = Image.open('../input/johar-pic/79669868_10158168171086929_2402259235392978944_n.jpg')
# report the size of the image
print(image.size)
# resize image and ignore original aspect ratio
img_resized = image.resize((200,200))
# report the size of the thumbnail
print(img_resized.size)
img_resized 