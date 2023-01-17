# Load training data

import numpy as np

import pandas as pd

import pyarrow.parquet as pq

from tqdm import tqdm_notebook as tqdm



v_raw_train = pq.read_pandas('../input/train.parquet').to_pandas().values

meta_train = np.loadtxt('../input/metadata_train.csv', skiprows=1, delimiter=',')

y_train = meta_train[:, 3].astype(bool)



print(v_raw_train.shape)
def compute_spectra(v_raw, *, m = 1000):

    """

    compute mean and percentile - mean for every chunk of m data

    

    Args:

      v_raw (array): 800,000 x n_sample; the input

      m (int): the chunk size

    

    Returns: d (dict)

      d['mean']: mean in each chunks

      d['percentile']: percentile - mean

    """

    percentile = (100, 99, 95, 0, 1, 5)

    

    n = v_raw.shape[1] # number of samples

    length = v_raw.shape[0] // m # 800,000 -> 800

    n_spectra = len(percentile)

    

    mean_signal = np.zeros((n, length), dtype='float32') # mean in each chunk

    percentile_spectra = np.zeros((n, length, n_spectra), dtype='float32')

    

    # compute spectra

    print('computing spectra...', flush=True)

    for i in tqdm(range(n)):

        v = v_raw[:, i].astype('float32').reshape(-1, m) / 128.0

        

        mean = np.mean(v, axis=1)        

        s = np.abs(np.percentile(v, percentile, axis=1) - mean)

        

        # subtract baseline

        h = np.percentile(s, 5.0)

        s = np.maximum(0.0, s - h)



        mean_signal[i, :] = mean

        percentile_spectra[i, :, :] = s.T

            

    d = {}

    d['mean'] = mean_signal

    d['percentile'] = percentile_spectra

    

    return d



spec_train = compute_spectra(v_raw_train)

print('done.')
import tensorflow as tf



def max_windowed(spec, *, width=150, stride=10):

    """

    Smooth the spectrum with a tophat window function and find the

    peak inteval that maximises the smoothed spectrum.

    

    Returns: d(dict)

      d['w'] (array): smoothed max - mean spectrum

      d['ibegin'] (array): the left edge index of the peak interval

    """

    n = spec.shape[0]

    length = spec.shape[1] # 800

    nspec = spec.shape[2] # 6 spectra



    n_triplet = n // 3



    # Reorganize the max spectrum from 8712 data to 2904 triplets with 3 phases

    max_spec3 = np.empty((n_triplet, length, 3))

    for i_triplet in range(n_triplet):

        max_spec3[i_triplet, :, 0] = spec[3*i_triplet, :, 0] # phase 0

        max_spec3[i_triplet, :, 1] = spec[3*i_triplet + 1, :, 0] # phase 1

        max_spec3[i_triplet, :, 2] = spec[3*i_triplet + 2, :, 0] # phase 2



    x = tf.placeholder(tf.float32, [None, length, 3]) # input spectra before smoothing

    # 800 -> 80: static convolaution

    # convolution but not CNN, the kernel is static

    # smoothing/convolution kernel

    # tophat window function

    # shape (3, 1) adds up 3 phases to one output

    K = np.ones((width, 3, 1), dtype='float32') / width



    W_conv1 = tf.constant(K)

    h_conv1 = tf.nn.conv1d(x, W_conv1, stride=stride, padding='VALID')

    

    with tf.Session() as sess:

        w = sess.run(h_conv1, feed_dict={x:max_spec3})



    imax = np.argmax(w[:, :, 0], axis=1) # index of maximum smoothed spectrum

    

    d = {}

    d['w'] = w # smoothed max spectrum

    d['ibegin'] = imax*stride

    

    return d



peaks = max_windowed(spec_train['percentile'])
def compute_features(v_raw, spec=None):

    """

    Args:

      v_raw (array): The original 800,000 x 8712 training data

      spec (dict): The result of compute_spectra() if already computed.

                   If it is None, it will be computed automatically.

    

    Returns:

       X (array): Feature vector of shape (2904, 57)

                  2904 triplets, 57 features

    """

    if spec is None:

        spec = compute_spectra(v_raw)

    

    v_spec = spec['percentile']

    shape = v_spec.shape

    n = shape[0] # number of data

    length = shape[1]

    nspec = shape[2]

    

    n_triplet = n // 3

    

    # Reorder to i_triplet, phase

    spec3 = np.empty((n_triplet, length, nspec, 3))

    

    for i_triplet in range(n_triplet):

        spec3[i_triplet, :, :, 0] = v_spec[3*i_triplet, :, :] # phase 0

        spec3[i_triplet, :, :, 1] = v_spec[3*i_triplet + 1, :, :] # phase 1

        spec3[i_triplet, :, :, 2] = v_spec[3*i_triplet + 2, :, :] # phase 2



    # extract "max-windowed" from the spectra

    width = 150

    peaks = max_windowed(v_spec, width=width)

    

    # Feature vector

    n_feature4 = 3

    X = np.empty((n_triplet, n_feature4*nspec*3 + 3))

    

    # features for each percentile and phase

    X4 = np.empty((n_triplet, n_feature4, nspec, 3)) # triplet, figure, spec type, phase

        

    for i_triplet in range(n_triplet):       

        # Maximum of the spectra in the full range

        # 18 features (6 percentiles x 3 phases)

        X4[i_triplet, 0, :, :] = np.max(spec3[i_triplet, :, :, :], axis=0)

        

        # Peak interval

        ibegin = peaks['ibegin'][i_triplet]

        iend = ibegin + width

        imid = ibegin + width // 2

    

        # Mean of the spectra in the peak inteval 18 features

        X4[i_triplet, 1, :, :] = np.mean(spec3[i_triplet, ibegin:iend, :, :], axis=0)

        

        # Max of the spectra in the peak inteval (18 features)

        X4[i_triplet, 2, :, :] = np.max(spec3[i_triplet, ibegin:iend, :, :], axis=0)

        

        # Mean signal at the midpoint of the interval (3 features)

        X[i_triplet, 0] = spec['mean'][3*i_triplet,     imid]

        X[i_triplet, 1] = spec['mean'][3*i_triplet + 1, imid]

        X[i_triplet, 2] = spec['mean'][3*i_triplet + 2, imid]

    

    shape = X4.shape

    

    # Flatten the X4 tensor

    # 3 + 18x3 = 57 features

    X[:, 3:] = X4.reshape(shape[0], shape[1]*shape[2]*shape[3])

    

    return X



X_all3 = compute_features(v_raw_train, spec_train)



# The label for the triple

# True iff two or more labels in 3 phases are True

y_all3 = np.sum(y_train.reshape(-1, 3), axis=1) >= 2



print('Three phases are combined into one training data; the shapes are, therefore,')

print(X_all3.shape, y_all3.shape)
train = pd.DataFrame(data=X_all3, columns=['col_'+str(i) for i in range(57)])
train.head()
train['target'] = y_all3*1
train.head()
# Release RAM of the training data 

if 'v_raw_train' in globals():

    del v_raw_train



# Load test data

id_test = np.loadtxt('../input/metadata_test.csv', skiprows=1, delimiter=',')[:, 0].astype(int)

n_test = len(id_test)



X_tests = []



# Load test data and compute the feature vector

# The test data is split into 4 to fit it into RAM

n_subset = 4

nread = 0



for i_subset in range(n_subset):

    # signal_id range in the test data; 8712 is the first data in the test.parquet

    ibegin = 8712 + 3*int(n_test // 3 * (i_subset/n_subset))

    iend = 8712 + 3*int(n_test // 3 * ((i_subset + 1)/n_subset))

    

    print('Loading %d/%d; signal_id %d - %d...' % (i_subset, n_subset, ibegin, iend))

    v_raw_test = pq.read_pandas('../input/test.parquet',

                                columns=[str(i) for i in range(ibegin, iend)]).to_pandas().values

    

    nread += v_raw_test.shape[1]

    X = compute_features(v_raw_test)

    X_tests.append(X)

    print('%d/%d test data processed.' % (nread, n_test))



    del v_raw_test



X_test = np.concatenate(X_tests, axis=0)

assert(X_test.shape[0] == id_test.shape[0] // 3)



del X_tests



print('X_test computation done. shape', X_test.shape)
test = pd.DataFrame(data=X_test, columns=['col_'+str(i) for i in range(57)])
train.to_csv('train.csv', index=False)

test.to_csv('test.csv', index=False)