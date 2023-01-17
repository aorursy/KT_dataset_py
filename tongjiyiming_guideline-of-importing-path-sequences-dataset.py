import numpy as np

import scipy.sparse as sp
### input npz format. This is the way to bypass "ValueError: b'csr'".

train_sparse = np.load("../input/train_path_sequence.npz")

test_sparse = np.load("../input/test_path_sequence.npz")

print(train_sparse)

print(test_sparse)

train_sparse = sp.csr_matrix((train_sparse.f.data, 

                                     train_sparse.f.indices, 

                                     train_sparse.f.indptr

                                    ), shape=(1458644, 141505))

print(train_sparse.shape)

test_sparse = sp.csr_matrix((test_sparse.f.data, 

                                     test_sparse.f.indices, 

                                     test_sparse.f.indptr

                                    ), shape=(625134, 141505))

print(test_sparse.shape)
### simple way to import, if you have a Mac, or you have luck

# train_sparse = sp.sparse.load_npz("../input/train_path_sequence.npz")

# test_sparse = sp.sparse.load_npz("../input/test_path_sequence.npz")