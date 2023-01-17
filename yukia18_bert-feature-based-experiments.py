!pip install tensorflow_datasets
import os

import random



import numpy as np



from tqdm.notebook import tqdm



from sklearn.model_selection import StratifiedKFold

from sklearn.linear_model import RidgeClassifierCV

from sklearn.metrics import accuracy_score



import tensorflow as tf

import tensorflow_datasets



import torch



from transformers import BertModel, BertTokenizer
def seed_everything(seed=42):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True





def load_nlp_data(name):

    data = tensorflow_datasets.load(name)

    train = tuple(data['train'].as_numpy_iterator())

    valid = tuple(data['validation'].as_numpy_iterator())

    test  = tuple(data['test'].as_numpy_iterator())

    return train, valid, test





def load_vector_data(data, model, tokenizer, use='head'):

    assert use in ['head', 'mean', 'mean-no-head']

    

    X, y = list(), list()

    for sample in tqdm(data):

        # single sentence

        if 'sentence' in sample.keys():

            sentence = str(sample['sentence'])

            input_ids = torch.tensor([tokenizer.encode(sentence, add_special_tokens=True)])

        # double sentence

        elif 'sentence1' in sample.keys() and 'sentence2' in sample.keys():

            sentence1 = str(sample['sentence1'])

            sentence2 = str(sample['sentence2'])

            input_ids = torch.tensor([tokenizer.encode(sentence1, sentence2, add_special_tokens=True)])

        

        with torch.no_grad():

            x = model(input_ids)[0].numpy()

        

        # use [class] output

        if use == 'head':

            x = x[0,0,:]

        # use all token outputs mean

        elif use == 'mean':

            x = x[0,:].mean(axis=0)

        elif use == 'mean-no-head':

            x = x[0,1:].mean(axis=0)

        

        # append

        X.append(x)

        y.append(sample['label'])

    

    return np.array(X), np.array(y)
def nlp_test(

    name,

    model,

    tokenizer,

    use='head',

    n_splits=5,

    random_state=42,

    shuffle=True,

    alphas=(0.1, 1.0, 10.0),

    fit_intercept=True,

    normalize=True,

):

    train, valid, _ = load_nlp_data(name)

    

    X_train, y_train = load_vector_data(train, model, tokenizer, use=use)

    X_valid, y_valid = load_vector_data(valid, model, tokenizer, use=use)

    

    kf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=shuffle)

    cv = kf.split(X_train, y_train)

    clf = RidgeClassifierCV(alphas=alphas, fit_intercept=fit_intercept, normalize=normalize, cv=cv)

    clf.fit(X_train, y_train)

    

    train_score = clf.score(X_train, y_train)

    valid_score = accuracy_score(clf.predict(X_valid), y_valid)

    

    print('alpha: {:.5f}'.format(clf.alpha_))

    print('train ACC score: {:.5f}'.format(train_score))

    print('valid ACC score: {:.5f}'.format(valid_score))
N_SPLITS = 5

SEED = 42

ALPHAS = (0.1, 1.0, 10.0)



seed_everything(SEED)



model = BertModel.from_pretrained('bert-base-uncased')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
nlp_test('glue/cola', model, tokenizer, use='head', n_splits=N_SPLITS, random_state=SEED, alphas=ALPHAS)
nlp_test('glue/cola', model, tokenizer, use='mean-no-head', n_splits=N_SPLITS, random_state=SEED, alphas=ALPHAS)
nlp_test('glue/cola', model, tokenizer, use='mean', n_splits=N_SPLITS, random_state=SEED, alphas=ALPHAS)
nlp_test('glue/mrpc', model, tokenizer, use='head', n_splits=N_SPLITS, random_state=SEED, alphas=ALPHAS)
nlp_test('glue/mrpc', model, tokenizer, use='mean-no-head', n_splits=N_SPLITS, random_state=SEED, alphas=ALPHAS)
nlp_test('glue/mrpc', model, tokenizer, use='mean', n_splits=N_SPLITS, random_state=SEED, alphas=ALPHAS)