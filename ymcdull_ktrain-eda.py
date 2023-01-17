!pip install ktrain
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.datasets import fetch_20newsgroups



categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']

train_b = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)

test_b = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)



x_train = train_b.data

y_train = train_b.target

x_test  = test_b.data

y_test  = test_b.target



## To make the EDA faster

x_train = x_train[:300]

y_train = y_train[:300]

x_test  = x_test[:300]

y_test  = y_test[:300]
import ktrain

from ktrain import text
MODEL_NAME = 'bert-base-uncased'

t = text.Transformer(MODEL_NAME, maxlen=500, classes=list(set(y_train)))

trn = t.preprocess_train(x_train, y_train)

val = t.preprocess_test(x_test, y_test)
model = t.get_classifier()

learner = ktrain.get_learner(model, train_data=trn, val_data=val, batch_size=6)
learner.fit_onecycle(5e-5, 4)
predictor = ktrain.get_predictor(learner.model, preproc=t)
predictor.predict('Jesus Christ is the central figure of Christianity.')
!pip install git+https://github.com/amaiya/eli5@tfkeras_0_10_1
predictor.explain('Jesus Christ is the central figure of Christianity.')