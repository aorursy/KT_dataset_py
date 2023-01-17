import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
%matplotlib inline

from scipy import stats, linalg

import seaborn as sns

mydata = pd.read_csv("../input/kc_house_data.csv", parse_dates = ['date'])

#make a table of the data!

#categorize/clear the data

#zip codes are strings
mydata['zipcode'] = mydata['zipcode'].astype(str)
#the other non ratio data are "categories" thanks pandas :D
mydata['waterfront'] = mydata['waterfront'].astype('category',ordered=True)
mydata['condition'] = mydata['condition'].astype('category',ordered=True)
mydata['view'] = mydata['view'].astype('category',ordered=True)
mydata['grade'] = mydata['grade'].astype('category',ordered=False)

#drop ID
mydata = mydata.drop(['id', 'date'],axis=1)
mydata = mydata.dropna()
mydata = mydata[mydata.bedrooms < 15]
#display a table of all the data for refernce (handy)
df = pd.DataFrame(data = mydata)
str_list = [] # empty list to contain columns with strings (words)
for colname, colvalue in mydata.iteritems():
    if type(colvalue[1]) == str:
         str_list.append(colname)
# Get to the numeric columns by inversion            
num_list = mydata.columns.difference(str_list) 
# Create Dataframe containing only numerical features
numData = mydata[num_list]

#and then remove more stuff
interestingCol =['price','bedrooms','bathrooms','sqft_above','sqft_living']
numData = numData[interestingCol]
originalData = numData.copy()

#reduce the number of data points
numData = numData.sample(n=11000, random_state = 13)
originalData = numData.copy()

from sklearn.preprocessing import MinMaxScaler
#figure out what the standardized million dollars is and save that
oneMillSTD = (numData['price'].median()-numData['price'].mean())/numData['price'].std()
numData =(numData - numData.mean()) / numData.std()
numData.fillna(method='backfill', inplace=True)

numData.describe()
numData.head()
sns.set_context("paper")
sns.distplot(originalData['price'])

X = numData.drop(['price'],axis=1)
y = numData['price']

attributeNames = list(X)

classNames = ['MillionDollarHome','notMDH']

from sklearn import model_selection
X, X_testglb, y, y_testglb = model_selection.train_test_split(X,y, test_size = (1/11),random_state = 42)

N, M = X.shape

#Then we give it classes
def millionDollars(money):
    #returns false if the price is less than a million
    #returns true if the price is equal to or greater than a million dollars
    if(money < oneMillSTD):
        return 0
    else:
        return 1

#create the new classification data set
y_cat = y.apply(millionDollars)


scaler = MinMaxScaler()

Xmat = scaler.fit_transform(X)

Y = Xmat - np.ones((Xmat.shape[0],1))*Xmat.mean(0)
U,S,V = linalg.svd(Y,full_matrices=False)
V=V.T
#calculate the variance from principle components
rho = (S*S)/(S*S).sum()

cumsumrho = np.cumsum(rho)


reducedData = Xmat.dot(V)

princCompX1 = [[0],[0]]
princCompY1 = [[0],[1]]

princCompX2 = [[0],[1]]
princCompY2 = [[0],[0]]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(Xmat[:,1],Xmat[:,2], c=y_cat, marker='.')
#ax.plot(princCompX1,princCompY1, c='r')
#ax.plot(princCompX2,princCompY2, c='r')

plt.title('Data plotted along two PCA components')
plt.xlabel('PCA1')
plt.ylabel('PCA2')

plt.show()
from sklearn import mixture
import itertools

Xgmm = Xmat

lowest_bic = np.infty
bic = []
n_components_range = range(1, 6)
cv_types = ['tied']
for cv_type in cv_types:
    for n_components in n_components_range:
        # Fit a Gaussian mixture with EM
        gmm = mixture.GaussianMixture(n_components=n_components,
                                      covariance_type=cv_type)
        gmm.fit(Xgmm)
        bic.append(gmm.bic(Xgmm))
        if n_components == 5:
            bic[-1] = bic[-2]+1000
        
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm

bic = np.array(bic)
color_iter = itertools.cycle(['cornflowerblue'])
clf = best_gmm
bars = []

# Plot the BIC scores
spl = plt.subplot(2, 1, 1)
for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
    xpos = np.array(n_components_range) + .2 * (i - 2)
    bars.append(plt.bar(xpos, bic[i * len(n_components_range):
                                  (i + 1) * len(n_components_range)],
                        width=.2, color=color))
plt.xticks(n_components_range)
plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
plt.title('BIC score per model')
xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +.2 * np.floor(bic.argmin() / len(n_components_range))
plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
spl.set_xlabel('Number of components')
K = 3

from sklearn.mixture import GaussianMixture
cov_type = 'diag'       
# type of covariance, you can try out 'diag' as well
reps = 5                
# number of fits with different initalizations, best result will be kept
# Fit Gaussian mixture model
gmm = GaussianMixture(n_components=K, covariance_type=cov_type, n_init=reps).fit(X)
cls = gmm.predict(X)    
# extract cluster labels
cds = gmm.means_        
# extract cluster centroids (means of gaussians)
covs = gmm.covariances_
# extract cluster shapes (covariances of gaussians)
if cov_type == 'diag':    
    new_covs = np.zeros([K,M,M])    

count = 0    
for elem in covs:        
    temp_m = np.zeros([M,M])        
    for i in range(len(elem)):            
        temp_m[i][i] = elem[i]        
    
    new_covs[count] = temp_m        
    count += 1
        
covs = new_covs
# Plot results:

from matplotlib.patches import Ellipse

def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()
    
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    
    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))
        
def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
    ax.axis('equal')
    plt.title('GMM with K=3 clusters')
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)
plot_gmm(gmm, Xmat[:,1:3])

def plotDendro(Z,method):
    plt.figure(figsize=(25, 10))
    plt.title('Hierarchical Clustering Dendrogram Using the '+method+ 'Method')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    dendrogram(
    Z,
    truncate_mode='lastp',  # show only the last p merged clusters
    p=20,  # show only the last p merged clusters
    show_leaf_counts=False,  # otherwise numbers in brackets are counts
    leaf_rotation=90.,
    leaf_font_size=8.,
    show_contracted=True,  # to get a distribution impression in truncated branches
    )
    plt.show()
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist

Method = 'single'

Methods = ['single','complete','average','ward']

cophenetScore =[]
for Method in Methods:
    Z = linkage(X, method=Method)
    c, coph_dists = cophenet(Z, pdist(X))
    print(Method)
    cophenetScore.append(c)
    print("%.2f" %c)
    #plotDendro(Z,Method)

#too much work to recreate the plotting

measures = ['cityblock', 'cosine', 'jaccard', 'mahalanobis']

cophenetScore =[]
for measure in measures:
    Z = linkage(X, method='average',metric=measure)
    c, coph_dists = cophenet(Z, pdist(X, metric=measure))
    print(measure)
    cophenetScore.append(c)
    print("%.2f" %c)
    #plotDendro(Z,Method)

Z = linkage(X, method='average',metric='cosine')
c, coph_dists = cophenet(Z, pdist(X, metric='cosine'))
print(measure)
cophenetScore.append(c)
print("%.2f" %c)
#plotDendro(Z,Method)
def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata

fancy_dendrogram(
    Z,
    truncate_mode='lastp',
    p=12,
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,
    annotate_above=10,  # useful in small plots so annotations don't overlap
    max_d = .8,
)
plt.show()
from scipy.cluster.hierarchy import fcluster
max_d = .8
clusters = fcluster(Z, max_d, criterion='distance')
plt.figure()
plt.title('Hierarchy grouping of data with 3 branches')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.scatter(Xmat[:,1], Xmat[:,2], c=clusters, s=40, cmap='viridis', zorder=2)  # plot points with cluster dependent colors
plt.show()
from sklearn import metrics
Hscore = metrics.adjusted_rand_score(y_cat,clusters)

labels = gmm.fit(Xmat).predict(Xmat)

GMMscore = metrics.adjusted_rand_score(y_cat,labels)
print("%.2f" %Hscore)
print("%.2f" %GMMscore)
print(type(labels))
