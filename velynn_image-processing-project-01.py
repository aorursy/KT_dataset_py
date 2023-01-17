# Importation des modules pertinent ici. 





# jugez nécessaires ici

import os

for dirname, _, filenames in os.walk('/kaggle/input'):





    for filename in filenames:

        print(os.path.join(dirname, filename))



# import module we'll need to import our custom module

from shutil import copyfile

copyfile(src = "../input/exemples/interpolation_bilineaire.png", dst = "../working/interpolation_bilineaire.png")

copyfile(src = "../input/exemples/interpolation_ppv.png", dst = "../working/interpolation_ppv.png")

copyfile(src = "../input/python/inf600f_tp1.py", dst = "../working/inf600f_tp1.py")

copyfile(src = "../input/imagesdata/Barbara.tif", dst = "../working/Barbara.tif")

copyfile(src = "../input/imagesdata/IRM_genou.tif", dst = "../working/IRM_genou.tif")

copyfile(src = "../input/imagesdata/Lune.tif", dst = "../working/Lune.tif")





from inf600f_tp1 import interpolation_bil

import matplotlib.pyplot as plt

import numpy as np

from skimage import data

import sys

import imageio

import scipy

import skimage

def interpolation_ppv(image, scale):

    """Fonction effectuant le changement d'échelle de l'image `image` selon le facteur `scale` 

       en utilisant l'interpolation par plus proche voisin.

   

   Paramètre(s) d'entrée

   -------------------

   image : ndarray

       Image (niveaux de gris) d'un type reconnu par Python.

   scale : float

       Paramètre de changement d'échelle. Un nombre réel strictement positif

       

   Paramètre(s) de sortie

   ----------------------

   image_p : ndarray

       Image interpolé àa la nouvelle échelle, de même type que `image`

   """

    # saving the type of the image

    dtype = image.dtype



    #Adding padding to the image to use it later for calculation

    img_p = np.pad(image.astype(np.float32), 1)



    # Calculation of the size of the original image and of the interpolated image

    height,width = image.shape 

    Scaled_width = (width * scale)

    Scaled_height = (height * scale)



    # Calculation of pixel coordinates in the interpolated image

    Scaled_X_coordinates=np.linspace(1, int(width), num=int(Scaled_width))

    Scaled_Y_coordinates=np.linspace(1, int(height), num=int(Scaled_height))



    #Initialization of the final image

    finalMatrix= np.zeros(shape=(int(Scaled_height) ,int(Scaled_width)))



    #rounding my scaled positions

    Scaled_X_coordinates=np.round(Scaled_X_coordinates)

    Scaled_Y_coordinates=np.round(Scaled_Y_coordinates)

    

    #Empty pixels array Initialization

    pixels=[]

    

    #Here, i store every pixels from the original image using the scaled coordinates

    #into an array of pixels

    for Line in Scaled_Y_coordinates.astype(int)  :

        for Column in Scaled_X_coordinates.astype(int):

            pixel = img_p[Line,Column]

            pixels.append(pixel)

            

    #Here i reconstruct the scaled image using the array of pixels from above

    Pixel_counter=0

    for i in range(int(Scaled_height)):

        for j in range(int(Scaled_width)):

            finalMatrix[i][j]=pixels[Pixel_counter]

            Pixel_counter=Pixel_counter+1



    #returning final matrix with the same type as the given img

    return finalMatrix.astype(dtype)
# Ajouter ici des cellules de code / markdown besoin pour vos réponses

img =  imageio.imread("./Barbara.tif").astype(dtype='float')

f = plt.figure(figsize=(10,5))

ax = f.add_subplot(131)

plt.title('Original Image', fontsize=10)

plt.imshow(img,cmap='gray')

plt.axis('off')

ax = f.add_subplot(132)

plt.title('Interpolation ppv (delta = 0.5)', fontsize=10)

New_Img_ppv=interpolation_ppv(img,0.5)

plt.imshow(New_Img_ppv,cmap='gray')

plt.axis('off')

ax = f.add_subplot(133)

plt.title('Interpolation ppv (delta  = 1/0.5)', fontsize=10)

FinalImageppv=interpolation_ppv(New_Img_ppv,1/0.5)

plt.imshow(FinalImageppv,cmap='gray')

plt.axis('off')

plt.show()
f = plt.figure(figsize=(10,5))



ax = f.add_subplot(121)

plt.title('Interpolation bil (delta  = 0.5)', fontsize=10)

New_Img_bil=interpolation_bil(img,0.5)

plt.imshow(New_Img_bil,cmap='gray')

plt.axis('off')

ax = f.add_subplot(122)

plt.title('Interpolation bil (delta  = 1/0.5)', fontsize=10)

FinalImgbil=interpolation_bil(New_Img_bil,1/0.5)

plt.imshow(FinalImgbil,cmap='gray')

plt.axis('off')

plt.show()

Err_Q_bil = ((img-FinalImgbil)**2).mean()

Err_Q_ppv = ((img-FinalImageppv)**2).mean()

print("Δ𝑄 avec l'interpolation bilinéaire: %.2f" % Err_Q_bil)

print("Δ𝑄 avec l'interpolation par plus proche voisin: %.2f" % Err_Q_ppv)

# show histogram

img =  imageio.imread("./Barbara.tif").astype(dtype="float")

print("Image Original")

plt.hist(img.ravel(), bins=120)

plt.xlabel('Intensité')

plt.ylabel('nombre de pixels')

plt.show()

f = plt.figure(figsize=(10,3))

ax = f.add_subplot(121)

plt.title('Image reproduite bil', fontsize=10)

plt.hist(FinalImgbil.ravel(), bins=120)

plt.xlabel('Intensité')

plt.ylabel('nombre de pixels')

ax2 = f.add_subplot(122)

plt.title('Image reproduite ppv', fontsize=10)

plt.hist(FinalImageppv.ravel(), bins=120)

plt.xlabel('Intensité')

plt.ylabel('nombre de pixels')



plt.show()
import scipy.ndimage



img =  imageio.imread("./IRM_genou.tif").astype(dtype='float')

noyau = np.ones((3,3)) / (9**2)

img_simple = scipy.ndimage.convolve(img, noyau)

plt.axis('off')

plt.imshow(img_simple,cmap='gray')
from scipy.ndimage import gaussian_filter

img_gauss = gaussian_filter(img, (3, 3))

plt.axis('off')

plt.imshow(img_gauss,cmap='gray' ); plt.show()

from scipy import ndimage

img_median = ndimage.median_filter(img, (7,7))

plt.axis('off')

plt.imshow(img_median,cmap='gray'); plt.show()

img =  imageio.imread("./IRM_genou.tif").astype(dtype='float')



f = plt.figure(figsize=(10,5))

Bruit = abs(img-img_simple)

ax = f.add_subplot(131)

plt.title('img - Moyennage simple', fontsize=10)

plt.axis('off')

plt.imshow(Bruit)







Bruit = abs(img-img_gauss)

ax = f.add_subplot(132)

plt.title('img - Moyennage gaussien', fontsize=10)

plt.axis('off')

plt.imshow(Bruit)





Bruit = abs(img-img_median)

ax = f.add_subplot(133)

plt.title('img - Moyennage médian', fontsize=10)

plt.axis('off')

plt.imshow(Bruit)

plt.show()



# Erreur quadratique moyenne.



Err_Q_Moy_simple = ((img-img_simple)**2).mean()

Err_Q_Moy_Gauss = ((img-img_gauss)**2).mean()

Err_Q_Moy_Median = ((img-img_median)**2).mean()



print("Δ𝑄 utilisant le filtre simple: %.2f" % Err_Q_Moy_simple)

print("Δ𝑄 utilisant le filtre guassien: %.2f" % Err_Q_Moy_Gauss)

print("Δ𝑄 utilisant le filtre median: %.2f" % Err_Q_Moy_Median)

img =  imageio.imread("./Lune.tif").astype(dtype='float')

plt.axis('off')

plt.imshow(img,cmap='gray'); plt.show()
img =  imageio.imread("./Lune.tif").astype(dtype='float')

#Correction logarithmique

f = plt.figure(figsize=(10,10))

ax = f.add_subplot(121)

c = 255/(np.log(1 + np.max(img))) 

img_transformed = c * np.log(1 + img) 

plt.title('Correction logarithmique', fontsize=10)

plt.axis('off')

plt.imshow(img_transformed,cmap='gray'); 



#Correction Gamma appliqué par dessus la correction log

ax = f.add_subplot(122)

gamma=1.5

gamma_corrected = np.array(255*(img_transformed / 255) ** gamma, dtype = 'float')

plt.title('+ Correction gamma', fontsize=10)

plt.axis('off')

plt.imshow(gamma_corrected,cmap='gray'); plt.show()
def masque_flou(image, a=1, b=3):

    """Filtre de type `masque flou`

    

    Paramètres

    ----------

    image : ndarray

        Image à rehausser

    a : float

        Coefficient de rehaussement

    b : int

        Taille du filtre adoucisseur (sigma de la gaussienne)

    

    Sortie

    ------

    image_p : ndarray

        Image rehaussée    

    """

    image_flou = gaussian_filter(image, (b, b))

    image_p = image + a*(image - image_flou)

    

    return image_p.astype(dtype='float')
f = plt.figure(figsize=(10,10))

ax = f.add_subplot(121)



plt.title('Image without Unsharp filter', fontsize=10)

plt.axis('off')

plt.imshow(gamma_corrected,cmap='gray')

ax = f.add_subplot(122)

plt.title('With Unsharp filter(a=1,b=3)', fontsize=10)

plt.axis('off')

img_unsharpened = masque_flou(gamma_corrected,1,3)

plt.imshow(img_unsharpened,cmap='gray'); plt.show()



f = plt.figure(figsize=(10,10))

print("Differentes variations de a et b")

ax = f.add_subplot(131)

plt.title('With Unsharp filter(a=10,b=3)', fontsize=10)

img_unsharpened = masque_flou(gamma_corrected,10,3)

plt.axis('off')

plt.imshow(img_unsharpened,cmap='gray')



ax = f.add_subplot(132)

plt.title('With Unsharp filter(a=1,b=10)', fontsize=10)

img_unsharpened = masque_flou(gamma_corrected,1,10)

plt.axis('off')

plt.imshow(img_unsharpened,cmap='gray')



ax = f.add_subplot(133)

plt.title('With Unsharp filter(a=5,b=15)', fontsize=10)

img_unsharpened = masque_flou(gamma_corrected,5,15)

plt.axis('off')

plt.imshow(img_unsharpened,cmap='gray'); plt.show()

from skimage import exposure

img_unsharpened = masque_flou(gamma_corrected,1,3)

img_equalized = (exposure.equalize_hist(img_unsharpened))



print("Histogramme")

f = plt.figure(figsize=(10,5))



ax = f.add_subplot(121)

plt.hist(img_unsharpened.ravel(), bins=10)

plt.axis('off')

ax = f.add_subplot(122)

plt.hist(img_equalized.ravel(), bins=10)

plt.axis('off')

plt.show()



print("Resultats")

f = plt.figure(figsize=(10,5))

plt.subplot(121)

plt.title('With Unsharp filter(a=1,b=3)', fontsize=10)

plt.imshow(img_unsharpened, cmap='gray')

plt.axis('off')

plt.subplot(122)

plt.title('After Equalization', fontsize=10)

plt.imshow(img_equalized, cmap='gray')

plt.axis('off')

plt.tight_layout()

plt.show()

img =  imageio.imread("./Lune.tif").astype(dtype='float')

img_equalized = (exposure.equalize_hist(img))





f = plt.figure(figsize=(10,5))

plt.subplot(121)

plt.imshow(img, cmap='gray')

plt.axis('off')

plt.subplot(122)

plt.imshow(img_equalized, cmap='gray')

plt.axis('off')

plt.tight_layout()

plt.show()
