import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
kdata = pd.read_csv('../input/Iris.csv', header=0)
kdata.columns
from itertools import cycle
print(dir(cycle))
cycol = cycle('bgrcmk')
for species, leaf in kdata.groupby('Species'):
    plt.scatter(x = leaf.SepalLengthCm, y = leaf.SepalWidthCm, label = species, color = cycol.__next__())

plt.title('Iris Sepal : Length vs Width')
plt.xlabel('Sepal Length cm')
plt.ylabel('Sepal Width cm')
plt.legend()
plt.grid(True)
plt.show()
