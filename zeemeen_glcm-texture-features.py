import numpy as np, pandas as pd, matplotlib.pyplot as plt
import tqdm
from skimage.io import imread

trainids = pd.read_csv('../input/train.csv')['id'].tolist()
def read_image(imgid):
    fn = '../input/train/images/{}.png'.format(imgid)
    return imread(fn)[...,0].astype(np.float32) / 255

def read_mask(imgid):
    fn = '../input/train/masks/{}.png'.format(imgid)
    return imread(fn).astype(np.uint8)    
from skimage.feature import greycomatrix, greycoprops
from multiprocessing import Pool

def glcm_props(patch):
    lf = []
    props = ['dissimilarity', 'contrast', 'homogeneity', 'energy', 'correlation']

    # left nearest neighbor
    glcm = greycomatrix(patch, [1], [0], 256, symmetric=True, normed=True)
    for f in props:
        lf.append( greycoprops(glcm, f)[0,0] )

    # upper nearest neighbor
    glcm = greycomatrix(patch, [1], [np.pi/2], 256, symmetric=True, normed=True)
    for f in props:
        lf.append( greycoprops(glcm, f)[0,0] )
        
    return lf

def patch_gen(img, PAD=4):
    img1 = (img * 255).astype(np.uint8)

    W = 101
    imgx = np.zeros((101+PAD*2, 101+PAD*2), dtype=img1.dtype)
    imgx[PAD:W+PAD,PAD:W+PAD] = img1
    imgx[:PAD,  PAD:W+PAD] = img1[PAD:0:-1,:]
    imgx[-PAD:, PAD:W+PAD] = img1[W-1:-PAD-1:-1,:]
    imgx[:, :PAD ] = imgx[:, PAD*2:PAD:-1]
    imgx[:, -PAD:] = imgx[:, W+PAD-1:-PAD*2-1:-1]

    xx, yy = np.meshgrid(np.arange(0, W), np.arange(0, W))
    xx, yy = xx.flatten() + PAD, yy.flatten() + PAD

    for x, y in zip(xx, yy):
        patch = imgx[y-PAD:y+PAD+1, x-PAD:x+PAD+1]
        yield patch

def glcm_feature(img, verbose=False):
    
    W, NF, PAD = 101, 10, 4

    if img.sum() == 0:
        return np.zeros((W,W,NF), dtype=np.float32)
    
    l = []
    with Pool(3) as pool:
        for p in tqdm.tqdm(pool.imap(glcm_props, patch_gen(img, PAD)), total=W*W, disable=not verbose):
            l.append(p)
        
    fimg = np.array(l, dtype=np.float32).reshape(101, 101, -1)
    return fimg
def visualize_glcm(imgid):
    img = read_image(imgid)
    mask = read_mask(imgid)
    
    fimg = glcm_feature(img, verbose=1)
    
    _, (ax0, ax1) = plt.subplots(1, 2, figsize=(6,3))
    ax0.imshow(img)
    ax1.imshow(mask)
    plt.show()
    
    amin = np.amin(fimg, axis=(0,1))
    amax = np.amax(fimg, axis=(0,1))
    fimg = (fimg - amin) / (amax - amin)

    fimg[...,4] = np.power(fimg[...,4], 3)
    fimg[...,9] = np.power(fimg[...,9], 3)

    _, axs = plt.subplots(2, 5, figsize=(15,6))
    axs = axs.flatten()

    for k in range(fimg.shape[-1]):
        axs[k].imshow(fimg[...,k])
    plt.show()
visualize_glcm(np.random.choice(trainids))
visualize_glcm(np.random.choice(trainids))
visualize_glcm(np.random.choice(trainids))
visualize_glcm(np.random.choice(trainids))
visualize_glcm(np.random.choice(trainids))
visualize_glcm(np.random.choice(trainids))
visualize_glcm(np.random.choice(trainids))
