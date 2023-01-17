import matplotlib.pyplot as plt
import glob, os
    
for file in glob.glob("../input/train_sm/*.jpeg")[0:20]:
    im = plt.imread(file)
    plt.figure(figsize=(15,20))
    plt.imshow(im)
    plt.show()
