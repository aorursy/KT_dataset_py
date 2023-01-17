import numpy as np 
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt

from numpy.linalg import svd
        
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def compress(U, s, V, r):
    return U[:,:r].dot(np.diag(s)[:r, :r]).dot(V[:r,:])
        
img = mpimg.imread('/kaggle/input/lenna.png')

print("Original grayscaled image")

gray = rgb2gray(img)
plt.imshow(gray, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
plt.show()

U, s, V = svd(gray)

print("PCA: Graph of singular values")

plt.subplot(211)
plt.plot(s)
plt.subplot(212)
plt.plot(np.cumsum(s) / sum(s))
plt.show()
fig, ax = plt.subplots(1, 4)

for i, r in enumerate([10, 20, 50, 100]):
    compressed = compress(U, s, V, r)
    storage = U.shape[0] * U.shape[1] + V.shape[0] * V.shape[1] + len(s)
    compressed_storage = U.shape[0] * r + V.shape[0] * r + r

    ax[i].set_title(f"r = {r}\n{'{:2.1f}'.format(100.0 * compressed_storage / storage)}% of storage")
    ax[i].imshow(compressed, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)

gcf = plt.gcf()
size = gcf.get_size_inches()
gcf.set_size_inches(size[0]*2.5, size[1]*2.5, forward=True) # Set forward to True to resize window along with plot in figure.
plt.show()