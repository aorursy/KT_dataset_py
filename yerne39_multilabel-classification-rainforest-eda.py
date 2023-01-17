import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output

import matplotlib.pyplot as plt

from scipy.stats import bernoulli

import seaborn as sns

print(check_output(["ls", "../input"]).decode("utf8"))

#print(check_output(["ls", "../input/train-jpg"]).decode("utf8"))
sample = pd.read_csv('../input/sample_submission_v2.csv')

print(sample.shape)

sample.head()
df = pd.read_csv('../input/train_v2.csv')

df.head()
df.shape
all_tags = [item for sublist in list(df['tags'].apply(lambda row: row.split(" ")).values) for item in sublist]

print('total of {} non-unique tags in all training images'.format(len(all_tags)))

print('average number of labels per image {}'.format(1.0*len(all_tags)/df.shape[0]))
tags_counted_and_sorted = pd.DataFrame({'tag': all_tags}).groupby('tag').size().reset_index().sort_values(0, ascending=False)

tags_counted_and_sorted.head()
tags_counted_and_sorted.plot.barh(x='tag', y=0, figsize=(12,8))
tag_probas = tags_counted_and_sorted[0].values/tags_counted_and_sorted[0].values.sum()

indicators = np.hstack([bernoulli.rvs(p, 0, sample.shape[0]).reshape(sample.shape[0], 1) for p in tag_probas])

indicators = np.array(indicators)

indicators.shape
indicators[:10,:]
sorted_tags = tags_counted_and_sorted['tag'].values

all_test_tags = []

for index in range(indicators.shape[0]):

    all_test_tags.append(' '.join(list(sorted_tags[np.where(indicators[index, :] == 1)[0]])))

len(all_test_tags)
sample['tags'] = all_test_tags

sample.head()

sample.to_csv('bernoulli_submission.csv', index=False)
from glob import glob

image_paths = sorted(glob('../input/train-jpg/*.jpg'))[0:1000]

image_names = list(map(lambda row: row.split("/")[-1][:-4], image_paths))

image_names[0:10]
plt.figure(figsize=(12,8))

for i in range(6):

    plt.subplot(2,3,i+1)

    plt.imshow(plt.imread(image_paths[i]))

    plt.title(str(df[df.image_name == image_names[i]].tags.values))
from sklearn.multiclass import OneVsRestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV, train_test_split

from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler

from sklearn.metrics import fbeta_score, precision_score, make_scorer, average_precision_score

import cv2

import warnings



n_samples = 5000

rescaled_dim = 20



df['split_tags'] = df['tags'].map(lambda row: row.split(" "))

lb = MultiLabelBinarizer()

y = lb.fit_transform(df['split_tags'])

y = y[:n_samples]

X = np.squeeze(np.array([cv2.resize(plt.imread('../input/train-jpg/{}.jpg'.format(name)), (rescaled_dim, rescaled_dim), cv2.INTER_LINEAR).reshape(1, -1) for name in df.head(n_samples)['image_name'].values]))

X = MinMaxScaler().fit_transform(X)



print(X.shape, y.shape, lb.classes_)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)



clf = OneVsRestClassifier(LogisticRegression(C=10, penalty='l2'))



with warnings.catch_warnings():

    warnings.simplefilter('ignore')

    clf.fit(X_train, y_train)



score = fbeta_score(y_test, clf.predict(X_test), beta=2, average=None)

avg_sample_score = fbeta_score(y_test, clf.predict(X_test), beta=2, average='samples')

print('Average F2 test score {}'.format(avg_sample_score))

print('F2 test scores per tag:')

[(lb.classes_[l], score[l]) for l in score.argsort()[::-1]]
X_sub = np.squeeze(np.array([cv2.resize(plt.imread('../input/test-jpg/{}.jpg'.format(name)), (rescaled_dim, rescaled_dim), cv2.INTER_LINEAR).reshape(1, -1) for name in sample['image_name'].values]))

X_sub = MinMaxScaler().fit_transform(X_sub)

X_sub.shape
y_sub = clf.predict(X_sub)

all_test_tags = []

for index in range(y_sub.shape[0]):

    all_test_tags.append(' '.join(list(lb.classes_[np.where(y_sub[index, :] == 1)[0]])))

all_test_tags[0:20]
test_imgs = [plt.imread('../input/test-jpg/{}.jpg'.format(name)) for name in sample.head(20)['image_name'].values]

plt.imshow(test_imgs[7])
sample['tags'] = all_test_tags

sample.head()
sample.to_csv('ovr_f2_{}.csv'.format(avg_sample_score), index=False)
import cv2



n_imgs = 600



all_imgs = []



for i in range(n_imgs):

    img = plt.imread(image_paths[i])

    img = cv2.resize(img, (100, 100), cv2.INTER_LINEAR).astype('float')

#    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype('float')

    img = cv2.normalize(img, None, 0.0, 1.0, cv2.NORM_MINMAX)

    img = img.reshape(1, -1)

    all_imgs.append(img)



img_mat = np.vstack(all_imgs)

img_mat.shape
from scipy.spatial.distance import pdist, squareform



sq_dists = squareform(pdist(img_mat))

print(sq_dists.shape)

sns.clustermap(

    sq_dists,

    figsize=(12,12),

    cmap=plt.get_cmap('viridis')

)
%matplotlib inline

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls
from sklearn.manifold import TSNE

tsne = TSNE(

    n_components=3,

    init='random', # pca

    random_state=101,

    method='barnes_hut',

    n_iter=500,

    verbose=2

).fit_transform(img_mat)
trace1 = go.Scatter3d(

    x=tsne[:,0],

    y=tsne[:,1],

    z=tsne[:,2],

    mode='markers',

    marker=dict(

        sizemode='diameter',

        #color = preprocessing.LabelEncoder().fit_transform(all_image_types),

        #colorscale = 'Portland',

        #colorbar = dict(title = 'images'),

        line=dict(color='rgb(255, 255, 255)'),

        opacity=0.9

    )

)



data=[trace1]

layout=dict(height=800, width=800, title='3D embedding of images')

fig=dict(data=data, layout=layout)

py.iplot(fig, filename='3DBubble')
mask = np.zeros_like(sq_dists, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# upper triangle of matrix set to np.nan

sq_dists[np.triu_indices_from(mask)] = np.nan

sq_dists[0, 0] = np.nan



fig = plt.figure(figsize=(12,8))

# maximally dissimilar image

ax = fig.add_subplot(1,2,1)

maximally_dissimilar_image_idx = np.nanargmax(np.nanmean(sq_dists, axis=1))

plt.imshow(plt.imread(image_paths[maximally_dissimilar_image_idx]))

plt.title('maximally dissimilar')



# maximally similar image

ax = fig.add_subplot(1,2,2)

maximally_similar_image_idx = np.nanargmin(np.nanmean(sq_dists, axis=1))

plt.imshow(plt.imread(image_paths[maximally_similar_image_idx]))

plt.title('maximally similar')



# # now compute the mean image

#ax = fig.add_subplot(1,3,3)

#mean_img = gray_imgs_mat.mean(axis=0).reshape(rescaled_dim, rescaled_dim, 3)

#plt.imshow(cv2.normalize(mean_img, None, 0.0, 1.0, cv2.NORM_MINMAX))

#plt.title('mean image')
from sklearn.manifold import TSNE

tsne = TSNE(

    n_components=2,

    init='random', # pca

    random_state=101,

    method='barnes_hut',

    n_iter=500,

    verbose=2

).fit_transform(img_mat)
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def imscatter(x, y, images, ax=None, zoom=0.1):

    ax = plt.gca()

    images = [OffsetImage(image, zoom=zoom) for image in images]

    artists = []

    for x0, y0, im0 in zip(x, y, images):

        ab = AnnotationBbox(im0, (x0, y0), xycoords='data', frameon=False)

        artists.append(ax.add_artist(ab))

    ax.update_datalim(np.column_stack([x, y]))

    ax.autoscale()

    #return artists



nimgs = 500

plt.figure(figsize=(13,10))

imscatter(tsne[0:nimgs,0], tsne[0:nimgs,1], [plt.imread(image_paths[i]) for i in range(nimgs)])
from skimage import io

image_paths = sorted(glob('../input/train-tif/*.tif'))[0:1000]

imgs = [io.imread(path) / io.imread(path).max() for path in image_paths]

#r, g, b, nir = img[:, :, 0], img[:, :, 1], img[:, :, 2], img[:, :, 3]

ndvis = [(img[:,:,3] - img[:,:,0])/((img[:,:,3] + img[:,:,0])) for img in imgs]
plt.figure(figsize=(12,8))

plt.subplot(121)

plt.imshow(ndvis[2], cmap='jet')

plt.colorbar()

plt.title('NDVI index of cloudy image')

plt.subplot(122)

plt.imshow(imgs[32])
plt.figure(figsize=(12,8))

plt.subplot(121)

plt.imshow(ndvis[10], cmap='jet')

plt.colorbar()

plt.title('NDVI index of image with lots of vegetation')

plt.subplot(122)

plt.imshow(imgs[10])
import seaborn as sns

mndvis = np.nan_to_num([ndvi.mean() for ndvi in ndvis])

plt.figure(figsize=(12,8))

sns.distplot(mndvis)

plt.title('distribution of mean NDVIs')
sorted_idcs = np.argsort(mndvis)

print(len(sorted_idcs))

plt.figure(figsize=(12,8))

plt.subplot(221)

plt.imshow(imgs[sorted_idcs[0]])

plt.subplot(222)

plt.imshow(imgs[sorted_idcs[50]])

plt.subplot(223)

plt.imshow(imgs[sorted_idcs[-30]])

plt.subplot(224)

plt.imshow(imgs[sorted_idcs[-11]])
plt.imshow(imgs[sorted_idcs[500]])

plt.title('image with median NDVI value')