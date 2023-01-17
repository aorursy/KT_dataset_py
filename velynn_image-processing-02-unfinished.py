%load_ext autoreload

%autoreload 2



import os

for dirname, _, filenames in os.walk('/kaggle/input'):





    for filename in filenames:

        print(os.path.join(dirname, filename))



# Importation des modules pertinents ici. 

# Assurez-vous d'inclure tout autre module que vous

# jugez nécessaire ici



# import module we'll need to import our custom module

from shutil import copyfile

copyfile(src = "../input/python/inf600f_tp1.py", dst = "../working/inf600f_tp1.py")

copyfile(src = "../input/pythontp02/inf600f_tp2.py", dst = "../working/inf600f_tp2.py")

copyfile(src = "../input/images/images/astronaute.jpg", dst = "../working/astronaute.jpg")

copyfile(src = "../input/images/images/sur_exposition.png", dst = "../working/sur_exposition.png")

copyfile(src = "../input/images/images/masque_sous_exposition.png", dst = "../working/masque_sous_exposition.png")

copyfile(src = "../input/images/images/sous_exposition.png", dst = "../working/sous_exposition.png")

copyfile(src = "../input/images/images/sept_iles.png", dst = "../working/sept_iles.png")

copyfile(src = "../input/images/images/astronaute.jpg", dst = "../working/sept_iles.jpg")









import inf600f_tp2

import numpy as np

import matplotlib.pyplot as plt

import skimage

from scipy.fftpack import fft2, ifft2, fftshift, ifftshift, fftfreq



plt.rcParams["image.cmap"] = "gray" # Définit l'échelle de couleur "gray" en tant qu'échelle de couleur par défaut.

plt.rcParams["figure.dpi"] = 150 # Définit la résolution par défaut à 150 dpi
from inf600f_tp1 import interpolation_bil



def decomposition_pyramide_gauss(img, N):

    """ Décomposition d'une image en Pyramide de Gauss.

    Paramètre d'entrée

    ------------------

    img : ndarray

        Image à décomposer

    N : int

        Nombre de niveaux de décomposition



    Paramètre de sortie

    -------------------

    images : list

        Pyramide de Gauss sous forme d'une liste d'images



    Note : Pour la pyramide de Gauss, la première image dans la pyramide correspond 

           à l'image originale.

    """

    # creation du tableau representant les niveaux de decomposition (Niveau0->Image Initial)

    images = [img]

    

    # loop jusqu'a N nombre de niveaux

    for i in range(N):

        # Appliquation du filtre guassien sur l'image du niveau i.

        NextLevel = skimage.filters.gaussian(images[i], sigma=2) 

        # sous-échantillonnage d’un facteur 2 utilisant l'interpolation bilineaire

        NextLevel = interpolation_bil(NextLevel, 0.5)

        images.append(NextLevel)



    return images

from skimage import filters

img = inf600f_tp2.load_image_astronaute() # Création d'une image monochrome de taille 512x512

pyramide_gauss = decomposition_pyramide_gauss(img, N=5)

inf600f_tp2.display_pyramid(pyramide_gauss) # Visualisation de la pyramide
def decomposition_pyramide_laplace(img, N):

    """ Décomposition d'une image en Pyramide de Laplace.

    Paramètre d'entrée

    ------------------

    img : ndarray

        Image à décomposer

    N : int

        Nombre de niveaux de décomposition



    Paramètre de sortie

    -------------------

    images : list

        Pyramide de Laplace sous forme d'une liste d'images

    """

    # creation du tableau representant les niveaux de decomposition (Niveau0->Image Initial)

    images_floutee = [img]

    images= []

    

    # loop jusqu'a N nombre de niveaux

    for i in range(N):

        #Application du lissage gaussien

        Gaussian = skimage.filters.gaussian(images_floutee[i], sigma=2)

        #Sous echantillonnage

#         Image_SousEchantillon = interpolation_bil(Gaussian, 0.5)

        Image_SousEchantillon = Gaussian[0::2,0::2]

        #Sur-echantilonage

        #utilisation de resize ici

        Image_SurEchantillon=skimage.transform.resize(Image_SousEchantillon, (Image_SousEchantillon.shape[0] *2 , Image_SousEchantillon.shape[1]* 2),anti_aliasing=True)

#         Image_SurEchantillon = interpolation_bil(Image_SousEchantillon, 2)

        # Calcul du residus

        Residus = images_floutee[i] - Image_SurEchantillon

        #Sauvegarde de l'image floutee

        images_floutee.append(Image_SousEchantillon)

        #sauvegarde du residus

        images.append(Residus)

        

    

    #le dernier niveau d'une pyramide de Laplace correspond à l'image floue

    images.append(images_floutee[N])

        

        



    return images
img = inf600f_tp2.load_image_astronaute() # Création d'une image monochrome de taille 512x512

pyramide_laplace = decomposition_pyramide_laplace(img, N=5)

inf600f_tp2.display_pyramid(pyramide_laplace) # Visualisation de la pyramide
def reconstruction_pyramide_laplace(images):

    """ Reconstruction d'une image monochrome à partir d'une pyramide de Laplace.

    Paramètre d'entrée

    ------------------

    images : list

        Pyramide de Laplace sous forme d'une liste d'images



    Paramètre de sortie

    -------------------

    img : ndimage

        Image monochrome reconstruite

    """

    # Nombre de niveaux de décomposition

    N = len(images)

    

    # On commence avec le dernier niveau (l'image floue)

    img = images[-1]

    

    for i_level in reversed(range(0, N-1)):

        #utilisation de resize

        Img_Surechantillonee = skimage.transform.resize(img, (img.shape[0] *2 , img.shape[1]* 2),anti_aliasing=True)

#         Img_Surechantillonee = interpolation_bil(img, 2)

        img = Img_Surechantillonee + images[i_level]



    return img

# Votre code ici. Vous pouvez ajouter des cellules au besoin.

img_reconstruite = reconstruction_pyramide_laplace(pyramide_laplace)

plt.imshow(img_reconstruite,cmap='gray')

Δ𝑄 = ((img-img_reconstruite)**2).mean()

print("Δ𝑄 : " + str(Δ𝑄))
def fusion_pyramidale(img1, img2, masque, N):

    """Fusion d'images à l'aide de pyramides de Laplace et de Gauss.



    Paramètres d'entrées

    --------------------

    img1 : Image surexposée de taille MxN

    img2 : Image sous-exposée de taille MxN

    masque : Image binaire de taille MxN indiquant les régions sous-exposées à conserver (1) 



    Paramètre de sortie

    -------------------

    img_fusion : ndarray

        Image fusionnée

    """    

    # Décomposition de l'image surexposée en pyramide de Laplace sur N niveaux

    decomposition_laplace_ImgSurex = decomposition_pyramide_laplace(img1, N)

    

    # Décomposition de l'image sous-exposée en pyramide de Laplace sur N niveaux

    decomposition_laplace_ImgSousex = decomposition_pyramide_laplace(img2, N)



    

    # Décomposition de l'image binaire en pyramide de Gauss sur N niveaux

    Decomposition_Guass_Masque = decomposition_pyramide_gauss(masque, N)

    

    # Fusion des images à chaque niveau de la façon suivante : 𝐼𝑘=𝑖𝑚1𝑘∗𝑚𝑘+𝑖𝑚2𝑘∗(1−𝑚𝑘)

    img_fusion_par_niveau = []

    for k in range(0, N):

        I = decomposition_laplace_ImgSurex[k] * Decomposition_Guass_Masque[k] + decomposition_laplace_ImgSousex[k] * (1-Decomposition_Guass_Masque[k])

        img_fusion_par_niveau.append(I)

    

    #test

    inf600f_tp2.display_pyramid(img_fusion_par_niveau) # Visualisation de la pyramide





    

    # Reconstruction de la pyramide de Laplace fusionnée

    img_fusion=reconstruction_pyramide_laplace(img_fusion_par_niveau)



    

    return img_fusion
img1, img2, masque = inf600f_tp2.load_imgages_fusion() # img1 est l'image sur-exposée, img2 est l'image sous-exposée, et masque est une image binaire des régions sous-exposées à conserver.

img_fusion=fusion_pyramidale(img1, img2, masque, 7)

plt.imshow(img_fusion)

plt.show()



# plt.rcParams["image.cmap"] = "viridis" 

img = inf600f_tp2.load_image_astronaute()

plt.imshow(img)
from scipy.fftpack import fft2, fftshift

# calculer la transformée de Fourier

img_fft = fft2(img)

# placer la fréquence nulle au centre de l'image

img_fftshift = fftshift(img_fft)
def display_fft(img_fft):

    """Affichage de l'amplitude et de la phase d'une transformée de Fourier"""

    

    Amp = np.log(np.abs(fftshift(img_fft)))

    

    Phase = (np.angle(fftshift(img_fft)))

    

    print("Amplitude: ")

    plt.imshow(Amp,extent=[-0.5,.5,-0.5,.5]);plt.show()

    print("Phase: ")

    plt.imshow(Phase,extent=[-0.5,.5,-0.5,.5]);plt.show()

# Votre code et vos réponses ici. Ajoutez des cellules au besoin.

display_fft(img_fft)
img = inf600f_tp2.load_image_moire()

plt.imshow(img);plt.show()

# calculer la transformée de Fourier

img_fft = fft2(img)

display_fft(img_fft)
import math

img = inf600f_tp2.load_image_moire()



# Transformee de Fourrier



# L’algorithme FFT retourne le résultat de la transformée de 

# Fourier en donnant d’abord les fréquences positives en ordre 

# croissant, puis les fréquences négatives. 

img_fft = fft2(img)

# placer la fréquence nulle au centre de l'image

# On utilise fftshift pour réordonner le résultat

# de la transformée de Fourier pour s’assurer que 

# la fréquence nulle soit au centre. 

img_fft = fftshift(img_fft)



#Amplitude de la Transformee de Fourrier

Amplitude_img_fft = np.abs(img_fft)

# 1)



# Pour le premier masque, il faut calculer le percentile et appliquer l’applique à l’amplitude

# de la transformée de Fourier (np.abs(fft)) et non pas aux valeurs complexes.

# A représente la matrice Amplitude_img_fft. np.percentile(A, 90) nous retourne 

# un seul nombre, soit le 90e percentile. Pour essayer de comprendre, dis-toi 

# que c’est similaire au calcul de la moyenne. np.mean(A) retourne la moyenne 

# de toutes les amplitudes contenues dans la matrice A, alors que np.percentile(A) 

# retourne le 90e percentile de toutes les amplitudes contenues dans A.

mask = Amplitude_img_fft > np.percentile(Amplitude_img_fft, 90)



# Explication de ce qui est retourné dans mask:

# On obtient 2 valeurs parce que l’opération booléenne « plus grand que » (>)

# a été appliquée. Ici en arrière-plan numpy compare chaque élément de la

# matrice Amplitude_img_fft avec la valeur retournée par np.percentile. Si 

# l’élément de la matrice est plus grand que np.percentile(…), l’élément 

# correspondant dans mask prendra la valeur True, et au contraire si 

# Amplitude_img_fft[i,j] est plus petit que np.percentile(…), mask[i,j]

# prendra la valeur False.

plt.imshow(mask);plt.show()



# 2) 

# forumle de pyth.

def pythag(a, b):

    return math.sqrt(a*a + b*b)



# calucls des informations necessaires

height, width = mask.shape

height_center = math.floor(height/2)

width_center = math.floor(width/2)



# initialisation de ma matrices de distances



# Pour mettre à zéro les fréquences plus petites qu’un seuil de 1/sqrt(50), 

# il faut d’abord créer une matrice contenant pour chaque pixel la distance

# euclidienne entre le centre de la représentation de Fourier et chaque pixel.

# permet d’isoler les basses fréquences dans la représentation 

# de Fourier. Puisqu’on place la fréquence nulle au centre,

# plus on s’éloigne du centre et plus on représente des 

# fréquences élevées. La matrice de distance permet donc de 

# connaitre la distance par rapport au centre de la matrice 

# de Fourier, et on peut utiliser cette distance pour créer 

# des filtres passe-haut, passe-bas, etc.



DistMatrix = [ [ 0 for i in range(width) ] for j in range(height) ] 

DistMatrix=np.array(DistMatrix)



for i in range(0, width):

    for j in range(0, height):

      dist = pythag(i-width_center, j-height_center)

      DistMatrix[i][j]=dist



# conserver dans le masque toutes les fréquences spatiales inférieures à  1/50⎯⎯⎯⎯√ 

# La fréquence de coupure 1/sqrt(50) est donnée en termes de fréquence normalisée 

# (qui vont de -1/2 à 1/2 pour chaque dimension). Il faut donc que les distances soient

# également exprimées en termes de fréquence normalisée.



# freq = np.fft.fftfreq(DistMatrix.size, d=1) # je ne sais pas comment utiliser cette fonction

# Je ne comprend absolument pas cette partie sur la normalisation de frequences :/

mask2 = DistMatrix < (1 / math.sqrt(50)) 



# On veut utiliser mask2 pour s’assurer de conserver 

# les basses fréquences dans la FFT après le filtrage,

# parce que ce sont ces fréquences qui contiennent une 

# bonne partie de l’information nécessaire à la reconstruction 

# des objets dans l’image. On peut combiner les masques en 

# utilisant l’indexation booléenne disponible en numpy. 



# assigner la valeur de 0 à ces pixels dans l'image binaire



mask[mask2]=0

plt.imshow(mask);plt.show()



# 3)



# Pour obtenir l’inverse d’une matrice booléenne

mask_inv = ~mask



Correction = img_fft * (mask_inv)

display_fft(Correction)

# 4)

img_p = ifft2(ifftshift(Correction))

plt.subplot(121); plt.imshow(img_p.real); plt.title("Partie Réelle"); plt.colorbar()

plt.subplot(122); plt.imshow(img_p.imag); plt.title("Partie imaginaire"); plt.colorbar()

plt.show()
