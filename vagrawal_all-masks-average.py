import numpy as np
import matplotlib.pyplot as plt
import glob
masks = np.array([plt.imread(img) for img in glob.glob("../input/train/*.tif") if 'mask' in img])
plt.imshow(masks.sum(axis=0))