import matplotlib.pyplot as plt

import glob, os

    

for file in glob.glob("../input/train/dog.*.jpg")[0:20]:

    im = plt.imread(file)

    plt.figure(figsize=(10,10))

    plt.imshow(im)

    plt.show()