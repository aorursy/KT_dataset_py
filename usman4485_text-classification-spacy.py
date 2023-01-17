# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

spam = pd.read_csv('../input/nlp-course/spam.csv')
spam
import spacy

#creating an empty model

nlp = spacy.blank('en')

textcat = nlp.create_pipe('textcat',

                          config={

                             'exclusive_classes':True,

                              'architecture':'bow'

                         })

nlp.add_pipe(textcat)
#adding labels to the model

textcat.add_label('spam')

textcat.add_label('ham')
train_text = spam['text'].values

train_labels = [{'cats':{'ham':label=='ham','spam':label=='spam'}} for label in spam['label']]
train_labels[0:5]
train_data = list(zip(train_text,train_labels))
train_data
from spacy.util import minibatch

spacy.util.fix_random_seed(1)

optimizer = nlp.begin_training()



batches = minibatch(train_data)

for batch in batches:

    text,labels = zip(*batch) # *zip(*) takes iterables as an argument

    nlp.update(text,labels,sgd=optimizer) #updating model parameter
import random

random.seed(1)



spacy.util.fix_random_seed(1)

optimizer = nlp.begin_training()

losses = {}

for epoch in range(20):

    random.shuffle(train_data)

    batches = minibatch(train_data,size=8)

    #creating batches of size 8

    for batch in batches:

        text,labels = zip(*batch)

        nlp.update(text,labels,sgd=optimizer,losses=losses)

    print(losses)
texts =["Are you ready for the tea party????? It's gonna be wild",

         "URGENT Reply to this message for GUARANTEED FREE TEA"]

doc = [nlp.tokenizer(t) for t in texts]

textcat = nlp.get_pipe('textcat')

scores, _ = textcat.predict(doc)

print(scores)