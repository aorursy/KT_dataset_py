import numpy as np # linear algebra

from sklearn.datasets import load_wine

import matplotlib.pyplot as plt

%matplotlib inline

figsize=(10,5)
X, y = load_wine(return_X_y=True)

print(f"X.shape={X.shape}, y.shape={y.shape}")
X_std = (X-X.mean(axis=0))/X.std(axis=0)
cov_mat = np.cov(X_std.T)

print(f"Covariance matrix shape:{cov_mat.shape}")
# Eigen decompose the covariance matrix

eigen_values, eigen_vectors = np.linalg.eig(cov_mat)

eigen_pairs = [(np.abs(eigen_values[i]), eigen_vectors[:,i]) for i in range(len(eigen_values))]
# Sort the eigen pairs by eigen values in descending order

eigen_pairs.sort(key=lambda k: k[0], reverse=True)
total_var = np.sum([i[0] for i in eigen_pairs])

expl_var = [i[0]/total_var for i in eigen_pairs]

cumulative_var = np.cumsum(expl_var)
plt.figure(figsize=figsize)

plt.bar(range(1,X.shape[1]+1), expl_var, alpha=0.5, align='center',label='Individual explained variance')

plt.step(range(1,X.shape[1]+1), cumulative_var, where='mid',label='Cumulative explained variance')

plt.ylabel('Explained variance ratio')

plt.xlabel('Principal component')

plt.legend(loc='best')

plt.grid();

plt.tight_layout();
w = np.hstack((eigen_pairs[0][1][:, np.newaxis],eigen_pairs[1][1][:, np.newaxis]))

print(f"Transformation matrix shape:{w.shape}")
X_pca = X_std.dot(w)

plt.figure(figsize=figsize)

colors =['tab:blue','tab:orange','tab:green']

markers=list('svo')

for l,c,m in zip(np.unique(y), colors, markers):

    plt.scatter(X_pca[y==l, 0],

                X_pca[y==l, 1],

                color='white',

                edgecolor=c,

                label=l,

                marker=m

               )

plt.xlabel('Principal Component 1')

plt.ylabel('Principal Component 2')

plt.grid();

plt.legend()

plt.tight_layout();