from __future__ import print_function, division
import numpy as np

# make a 3,3 array with ones in every position

np.ones((3,3))
a = np.arange(0, 25) # create a vector from 0 to 24

print('a',a)
b = a.reshape((5,5)) # shape the array into a 5x5 matrix instead of a vector

print('b_shape', b.shape) # each numpy array has a property called shape which gives you the dimensions

print('b',b)
import matplotlib.pyplot as plt

# we can now show all the functions in plt by using the dir command. 

# The ones relevant for us are subplot and imshow

print(','.join(dir(plt)))
%matplotlib inline

plt.plot(a)
plt.hist(a, bins = 5)
# since that was a really boring histogram, we can make a normal distribution with 1000 points

norm_dist = np.random.normal(0, 25, size=(1000))

plt.hist(norm_dist, bins = 10)
plt.imshow(b)
plt.imshow(b, interpolation = 'none')
from skimage import data

image = data.camera() # get the camera image from the scikit-image data

plt.imshow(image, 

           cmap=plt.cm.gray) # we use a grayscale ma
fig, (ax1, ax2) = plt.subplots(1,2, figsize = (10, 5)) # make a new subplot figure with 2 axes

ax1.imshow(image, cmap=plt.cm.jet) # we can use a different color mapping scheme like jet (matlab's default)

ax1.set_title('Jet Colored Image')

ax1.axis('off') # turn off the axes lines



img1_ax = ax2.imshow(image, cmap = 'RdBu') # you can also specifiy color maps by name

ax2.set_title('Red Blue Image')

ax2.set_aspect(2) # change the aspect ratio to 2:1

plt.colorbar(img1_ax) # show a color bar for the second image only
from skimage.filters import threshold_otsu





image = data.camera()

thresh = threshold_otsu(image)

binary = image > thresh



fig, axes = plt.subplots(ncols=3, figsize=(8, 2.5))

ax = axes.ravel()

ax[0] = plt.subplot(1, 3, 1, adjustable='box-forced')

ax[1] = plt.subplot(1, 3, 2)

ax[2] = plt.subplot(1, 3, 3, sharex=ax[0], sharey=ax[0], adjustable='box-forced')



ax[0].imshow(image, cmap=plt.cm.gray)

ax[0].set_title('Original')

ax[0].axis('off')



ax[1].hist(image.ravel(), bins=256)

ax[1].set_title('Histogram')

ax[1].axvline(thresh, color='r')



ax[2].imshow(binary, cmap=plt.cm.gray)

ax[2].set_title('Thresholded')

ax[2].axis('off')



plt.show()
try:

    from skimage.filters import try_all_threshold



    img = data.page()



    # Here, we specify a radius for local thresholding algorithms.

    # If it is not specified, only global algorithms are called.

    fig, ax = try_all_threshold(img, figsize=(10, 8), verbose=False)

    plt.show()

except ImportError:

    from warnings import warn

    warn('The current version of skimage does not support this feature, sorry', RuntimeWarning)
import os

from skimage.data import data_dir

from skimage.util import img_as_ubyte

from skimage import io



orig_phantom = img_as_ubyte(io.imread(os.path.join(data_dir, "phantom.png"),

                                      as_grey=True))

fig, ax = plt.subplots()

ax.imshow(orig_phantom, cmap=plt.cm.gray)
def plot_comparison(original, filtered, filter_name):



    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True,

                                   sharey=True)

    ax1.imshow(original, cmap=plt.cm.gray)

    ax1.set_title('original')

    ax1.axis('off')

    ax1.set_adjustable('box-forced')

    ax2.imshow(filtered, cmap=plt.cm.gray)

    ax2.set_title(filter_name)

    ax2.axis('off')

    ax2.set_adjustable('box-forced')
from skimage.morphology import erosion, dilation, opening, closing, white_tophat

from skimage.morphology import black_tophat 

from skimage.morphology import disk



selem = disk(6)

eroded = erosion(orig_phantom, selem)

plot_comparison(orig_phantom, eroded, 'erosion')
dilated = dilation(orig_phantom, selem)

plot_comparison(orig_phantom, dilated, 'dilation')
opened = opening(orig_phantom, selem)

plot_comparison(orig_phantom, opened, 'opening')
phantom = orig_phantom.copy()

phantom[10:30, 200:210] = 0



closed = closing(phantom, selem)

plot_comparison(phantom, closed, 'closing')
phantom = orig_phantom.copy()

phantom[340:350, 200:210] = 255

phantom[100:110, 200:210] = 0



w_tophat = white_tophat(phantom, selem)

plot_comparison(phantom, w_tophat, 'white tophat')
b_tophat = black_tophat(phantom, selem)

plot_comparison(phantom, b_tophat, 'black tophat')
from scipy import ndimage as ndi

from skimage import feature





# Generate noisy image of a square

im = np.zeros((128, 128))

im[32:-32, 32:-32] = 1



im = ndi.rotate(im, 15, mode='constant')

im = ndi.gaussian_filter(im, 4)

im += 0.2 * np.random.random(im.shape)



# Compute the Canny filter for two values of sigma

edges1 = feature.canny(im)

edges2 = feature.canny(im, sigma=3)



# display results

fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3),

                                    sharex=True, sharey=True)



ax1.imshow(im, cmap=plt.cm.gray)

ax1.axis('off')

ax1.set_title('noisy image', fontsize=20)



ax2.imshow(edges1, cmap=plt.cm.gray)

ax2.axis('off')

ax2.set_title('Canny filter, $\sigma=1$', fontsize=20)



ax3.imshow(edges2, cmap=plt.cm.gray)

ax3.axis('off')

ax3.set_title('Canny filter, $\sigma=3$', fontsize=20)



fig.tight_layout()



plt.show()
from scipy.ndimage import gaussian_filter

from skimage import img_as_float

from skimage.morphology import reconstruction



# Convert to float: Important for subtraction later which won't work with uint8

image = img_as_float(data.coins())

image = gaussian_filter(image, 1)



seed = np.copy(image)

seed[1:-1, 1:-1] = image.min()

mask = image



dilated = reconstruction(seed, mask, method='dilation')
fig, (ax0, ax1, ax2) = plt.subplots(nrows=1,

                                    ncols=3,

                                    figsize=(8, 2.5),

                                    sharex=True,

                                    sharey=True)



ax0.imshow(image, cmap='gray')

ax0.set_title('original image')

ax0.axis('off')

ax0.set_adjustable('box-forced')



ax1.imshow(dilated, vmin=image.min(), vmax=image.max(), cmap='gray')

ax1.set_title('dilated')

ax1.axis('off')

ax1.set_adjustable('box-forced')



ax2.imshow(image - dilated, cmap='gray')

ax2.set_title('image - dilated')

ax2.axis('off')

ax2.set_adjustable('box-forced')



fig.tight_layout()
h = 0.4

seed = image - h

dilated = reconstruction(seed, mask, method='dilation')

hdome = image - dilated
fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, figsize=(8, 2.5))

yslice = 197



ax0.plot(mask[yslice], '0.5', label='mask')

ax0.plot(seed[yslice], 'k', label='seed')

ax0.plot(dilated[yslice], 'r', label='dilated')

ax0.set_ylim(-0.2, 2)

ax0.set_title('image slice')

ax0.set_xticks([])

ax0.legend()



ax1.imshow(dilated, vmin=image.min(), vmax=image.max(), cmap='gray')

ax1.axhline(yslice, color='r', alpha=0.4)

ax1.set_title('dilated')

ax1.axis('off')



ax2.imshow(hdome, cmap='gray')

ax2.axhline(yslice, color='r', alpha=0.4)

ax2.set_title('image - dilated')

ax2.axis('off')



fig.tight_layout()

plt.show()
from skimage import measure





# Construct some test data

x, y = np.ogrid[-np.pi:np.pi:100j, -np.pi:np.pi:100j]

r = np.sin(np.exp((np.sin(x)**3 + np.cos(y)**2)))



# Find contours at a constant value of 0.8

contours = measure.find_contours(r, 0.8)



# Display the image and plot all contours found

fig, ax = plt.subplots()

ax.imshow(r, interpolation='nearest', cmap=plt.cm.gray)



for n, contour in enumerate(contours):

    ax.plot(contour[:, 1], contour[:, 0], linewidth=2)



ax.axis('image')

ax.set_xticks([])

ax.set_yticks([])

plt.show()
from skimage.morphology import watershed

from skimage.feature import peak_local_max





# Generate an initial image with two overlapping circles

x, y = np.indices((80, 80))

x1, y1, x2, y2 = 28, 28, 44, 52

r1, r2 = 16, 20

mask_circle1 = (x - x1)**2 + (y - y1)**2 < r1**2

mask_circle2 = (x - x2)**2 + (y - y2)**2 < r2**2

image = np.logical_or(mask_circle1, mask_circle2)



# Now we want to separate the two objects in image

# Generate the markers as local maxima of the distance to the background

distance = ndi.distance_transform_edt(image)

local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)),

                            labels=image)

markers = ndi.label(local_maxi)[0]

labels = watershed(-distance, markers, mask=image)



fig, axes = plt.subplots(ncols=3, figsize=(9, 3), sharex=True, sharey=True,

                         subplot_kw={'adjustable': 'box-forced'})

ax = axes.ravel()



ax[0].imshow(image, cmap=plt.cm.gray, interpolation='nearest')

ax[0].set_title('Overlapping objects')

ax[1].imshow(-distance, cmap=plt.cm.gray, interpolation='nearest')

ax[1].set_title('Distances')

ax[2].imshow(labels, cmap=plt.cm.spectral, interpolation='nearest')

ax[2].set_title('Separated objects')



for a in ax:

    a.set_axis_off()



fig.tight_layout()

plt.show()
from skimage.filters import rank



image = img_as_ubyte(data.camera())



# denoise image

denoised = rank.median(image, disk(2))



# find continuous region (low gradient -

# where less than 10 for this image) --> markers

# disk(5) is used here to get a more smooth image

markers = rank.gradient(denoised, disk(5)) < 10

markers = ndi.label(markers)[0]



# local gradient (disk(2) is used to keep edges thin)

gradient = rank.gradient(denoised, disk(2))



# process the watershed

labels = watershed(gradient, markers)



# display results

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8), sharex=True, sharey=True, subplot_kw={'adjustable':'box-forced'})

ax = axes.ravel()



ax[0].imshow(image, cmap=plt.cm.gray, interpolation='nearest')

ax[0].set_title("Original")



ax[1].imshow(gradient, cmap=plt.cm.spectral, interpolation='nearest')

ax[1].set_title("Local Gradient")



ax[2].imshow(markers, cmap=plt.cm.spectral, interpolation='nearest')

ax[2].set_title("Markers")



ax[3].imshow(image, cmap=plt.cm.gray, interpolation='nearest')

ax[3].imshow(labels, cmap=plt.cm.spectral, interpolation='nearest', alpha=.7)

ax[3].set_title("Segmented")



for a in ax:

    a.axis('off')



fig.tight_layout()

plt.show()
from skimage.transform import radon, rescale



image = io.imread(data_dir + "/phantom.png", as_grey=True)

image = rescale(image, scale=0.4, mode='reflect')



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5))



ax1.set_title("Original")

ax1.imshow(image, cmap=plt.cm.Greys_r)



theta = np.linspace(0., 180., max(image.shape), endpoint=False)

sinogram = radon(image, theta=theta, circle=True)

ax2.set_title("Radon transform\n(Sinogram)")

ax2.set_xlabel("Projection angle (deg)")

ax2.set_ylabel("Projection position (pixels)")

ax2.imshow(sinogram, cmap=plt.cm.Greys_r,

           extent=(0, 180, 0, sinogram.shape[0]), aspect='auto')



fig.tight_layout()

plt.show()
from skimage.transform import iradon



reconstruction_fbp = iradon(sinogram, theta=theta, circle=True)

error = reconstruction_fbp - image

print('FBP rms reconstruction error: %.3g' % np.sqrt(np.mean(error**2)))



imkwargs = dict(vmin=-0.2, vmax=0.2)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5),

                               sharex=True, sharey=True,

                               subplot_kw={'adjustable': 'box-forced'})

ax1.set_title("Reconstruction\nFiltered back projection")

ax1.imshow(reconstruction_fbp, cmap=plt.cm.Greys_r)

ax2.set_title("Reconstruction error\nFiltered back projection")

ax2.imshow(reconstruction_fbp - image, cmap=plt.cm.Greys_r, **imkwargs)

plt.show()