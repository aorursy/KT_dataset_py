import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import glob, os
smjpegs = [f for f in glob.glob("../input/train_sm/*.jpeg")]

set175 = [smj for smj in smjpegs if "set175" in smj]
print(set175)

first = plt.imread('../input/train_sm/set175_1.jpeg')
dims = np.shape(first)
print(dims)
np.min(first), np.max(first)
plt.imshow(first)
pixel_matrix = np.reshape(first, (dims[0] * dims[1], dims[2]))
