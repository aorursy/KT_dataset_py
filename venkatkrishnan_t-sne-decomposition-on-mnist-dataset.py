import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import datasets
from sklearn.manifold import TSNE

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# data from sklearn datasets
data = datasets.fetch_openml('mnist_784', version=1, return_X_y=True)

# Extract data & target from the dataset
pixel_data, targets = data
targets = targets.astype(int)
print("Shape of Pixel data : {}".format(pixel_data.shape))
# Reshape the pixel data into 28x28
single_image = pixel_data[5, :].reshape(28,28)

plt.imshow(single_image, cmap='gray')
plt.title(f"Image of the text: {targets[5]}", fontsize=15)
plt.show()
# Object of tSNE
tsne = TSNE(n_components=2, random_state=42)

x_transformed = tsne.fit_transform(pixel_data[:3000, :]) # Data upto 3000 rows
# convert the transformed data into dataframe
tsne_df = pd.DataFrame(np.column_stack((x_transformed, targets[:3000])), columns=['X', 'Y', "Targets"])

tsne_df.loc[:, "Targets"] = tsne_df.Targets.astype(int)
tsne_df.head(10)
plt.figure(figsize=(10,8))

g = sns.FacetGrid(data=tsne_df, hue='Targets', height=8)

g.map(plt.scatter, 'X', 'Y').add_legend()

plt.show()
