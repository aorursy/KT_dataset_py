import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
!pip install git+https://github.com/WojciechMigda/PyTsetlini@b3c9dad
from pytsetlini import TsetlinMachineClassifier # Tsetlin Machine Classifier we will be training

# we will be using sklearn's pipelines with these two

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import FunctionTransformer
from functools import lru_cache





@lru_cache(maxsize=4096)

def _as_bits(x, nbits):

    fmt = '{0:0' + str(nbits) + 'b}'

    return np.array([int(c) for c in fmt.format(x)][-nbits:])
def _unpack_bits(a, nbits):

    if len(a.shape) > 2:

        raise ValueError("_unpack_bits: input array cannot have more than 2 dimensions, got {}".format(len(a.shape)))



    a = np.clip(a, 0, 2 ** nbits - 1)

    if nbits == 8:

        a_ = np.empty_like(a, dtype=np.uint8)

        np.rint(a, out=a_, casting='unsafe')

        rv = np.unpackbits(a_, axis=1)

        return rv

    else:

        a_ = np.empty_like(a, dtype=np.uint64)

        np.rint(a, out=a_, casting='unsafe')

        F = np.frompyfunc(_as_bits, 2, 1)

        rv = np.stack(F(a_.ravel(), nbits)).reshape(a.shape[0], -1)

        return rv
class Preprocessor(Pipeline):

    def __init__(self, nbits):

        nbits = int(nbits)

        if not (1 <= nbits <= 64):

            raise ValueError("Preprocessor: nbits is out of a valid range, {} vs [1, 64]".format(nbits))

        self.nbits = nbits

        super(type(self), self).__init__(steps=[

            ('unpacker', FunctionTransformer(_unpack_bits, validate=False, kw_args={'nbits': nbits})),

        ])

    pass
train_df = pd.read_csv('../input/digit-recognizer/train.csv', header=0, dtype='uint8')

print('Read train data: {}'.format(train_df.shape))



test_df = pd.read_csv('../input/digit-recognizer/test.csv', header=0, dtype='uint8')

print('Read test data: {}'.format(test_df.shape))



train_y = train_df.label.values

train_X = train_df.values[:, 1:]



test_X = test_df.values
train_X = train_X // 16

test_X = test_X // 16
pre = Preprocessor(nbits=4)

clf = Pipeline(

    steps=[

        ('preprocessor', pre),

        ('clf', TsetlinMachineClassifier(random_state=1234,

                                         clause_output_tile_size=64,

                                         number_of_pos_neg_clauses_per_label=500,

                                         number_of_states=127,

                                         boost_true_positive_feedback=1,

                                         n_jobs=2,

                                         s=10.,

                                         threshold=25)

        )

        ]

    )

print(clf)
clf.fit(train_X, train_y, clf__n_iter=150)
y_hat = clf.predict(test_X)
pred = pd.DataFrame(y_hat)

pred.index += 1

pred.index.name = 'ImageId'

pred.columns = ['Label']

pred.to_csv('results_150.csv', header=True)