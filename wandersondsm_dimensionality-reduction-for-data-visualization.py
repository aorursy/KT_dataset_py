# Load libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
% matplotlib inline

# Load dataset
dataset = pd.read_csv('../input/Iris.csv')

# Show pairplot
sns.pairplot(dataset, hue='Species')
# Do PCA
pca = PCA(2)
datasetPCA = pca.fit_transform(dataset[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']])
datasetPCA = pd.DataFrame({'pc1': datasetPCA[:, 0], 'pc2': datasetPCA[:, 1], 'class': dataset['Species']})

datasetPCA.head()
# 0.92 + 0.05 = 0.97
pca.explained_variance_ratio_
# Split dataset by class
setosa = datasetPCA[datasetPCA['class'] == 'Iris-setosa']
versicolor = datasetPCA[datasetPCA['class'] == 'Iris-versicolor']
virginica = datasetPCA[datasetPCA['class'] == 'Iris-virginica']
# Plot in 2D
plt.scatter(x=setosa['pc1'], y=setosa['pc2'])
plt.scatter(x=versicolor['pc1'], y=versicolor['pc2'])
plt.scatter(x=virginica['pc1'], y=virginica['pc2'])
plt.legend(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])