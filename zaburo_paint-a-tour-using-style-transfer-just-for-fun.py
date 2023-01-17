%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
!wget https://unsplash.com/photos/KBKHXjhVQVM/download?force=true -O reindeer.jpg
img = Image.open('reindeer.jpg')
img = img.resize((int(img.size[0] * 0.6), int(img.size[1] * 0.6)), Image.BICUBIC)
img.save('reindeer.jpg')
img
!git clone https://github.com/jvkersch/pyconcorde
!pip install -e ./pyconcorde
cities = pd.read_csv('../input/cities.csv')
xy_int = (cities[['X', 'Y']] * 1000).astype(np.int64)
with open('xy_int.csv', 'w') as fp:
    print(len(xy_int), file=fp)
    print(xy_int.to_csv(index=False, header=False, sep=' '), file=fp)
!./pyconcorde/build/concorde/LINKERN/linkern -s 1 -o lk.sol -N 2 xy_int.csv > /dev/null
order = []
with open('lk.sol', 'r') as fp:
    lines = fp.readlines()
order = [int(v.split(' ')[0]) for v in lines[1:]] + [0]
plt.figure(figsize=(15, 10))
xy = cities.loc[order, ['X', 'Y']].values
plt.plot(xy[:, 0], xy[:, 1], lw=1., ms=10, c='black')
plt.axis('equal')
plt.gca().set_axis_off()
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
plt.margins(0,0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
# download a fork of the repository to use recent chainer
!git clone -b use-new-api https://github.com/zaburo-ch/chainer-fast-neuralstyle
%run chainer-fast-neuralstyle/generate.py reindeer.jpg -m chainer-fast-neuralstyle/models/composition.model -o composition.png
%run chainer-fast-neuralstyle/generate.py reindeer.jpg -m chainer-fast-neuralstyle/models/seurat.model -o seurat.png
fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(111)
xy = cities.loc[order, ['X', 'Y']].values
poly = plt.Polygon(xy, fc='black')
ax.add_patch(poly)
plt.axis('equal')
plt.gca().set_axis_off()
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
plt.margins(0,0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.savefig('mask.png', bbox_inches='tight', pad_inches=0, dpi=150)
img1 = Image.open(f'composition.png')
mask = Image.open('mask.png').convert('L')
mask = np.asarray(mask.resize(img1.size, Image.BICUBIC)) >= 256 // 2
img1 = np.asarray(img1).copy()
img2 = Image.open(f'seurat.png')
img1[mask] = np.asarray(img2)[mask]
Image.fromarray(img1)
!rm -rf chainer-fast-neuralstyle
!rm -rf pyconcorde