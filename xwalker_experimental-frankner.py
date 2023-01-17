import sys
import os
sys.path.append("..") 

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import warnings
import pandas as pd
import numpy as np
from glob import glob
from frankner.preprocessing import ContextNER
from frankner.model import BiLSTM
from frankner.metrics import F1Metrics
from frankner.metrics import all_metrics
from frankner.metrics import all_metrics_fold
from frankner.utils import build_matrix_embeddings
from tcc_utils import save_best_model
from tcc_utils import save_stats_to_disk
from tcc_utils import pip_predict_metric
from seqeval.metrics import f1_score
from seqeval.metrics import recall_score
from seqeval.metrics import precision_score
warnings.filterwarnings('ignore')
train = pd.read_csv('../data/tcc/TRAIN_AUGMENTATION.csv')
test = pd.read_csv('../data/tcc/TEST.csv')
#building vocab
all_words_vocab = train.Word.to_list() + test.Word.to_list()
ner_train = ContextNER(train, all_words_vocab)
ner_test = ContextNER(test, all_words_vocab, max_len=ner_train.max_len)
twitter_wb = list(map(lambda x: x if x.find('twitter') != -1 else '', glob('../Embeddings/GloVe/*')))
twitter_wb
%%time

glove_tweet = \
build_matrix_embeddings(path='../Embeddings/GloVe/glove.twitter.27B.200d.txt',
                        num_tokens=ner_train.num_words, 
                        embedding_dim=200, 
                        word_index=ner_train.word2idx)

# glove = \
build_matrix_embeddings(path='../Embeddings/GloVe/glove.840B.300d.txt',
                        num_tokens=ner_train.num_words, 
                        embedding_dim=300, 
                        word_index=ner_train.word2idx)

fasttext = \
build_matrix_embeddings(path='../Embeddings/FastText/crawl-300d-2M.vec',
                        num_tokens=ner_train.num_words, 
                        embedding_dim=300, 
                        word_index=ner_train.word2idx)
wb_concatenate = np.concatenate([glove, fasttext, glove_tweet], axis=1)
wb_concatenate.shape
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
N_MODELS = 5
%%time

All_Historys = {}

for index_model in range(N_MODELS):
    
    file_name_model = '{}_model_crf_only_twitter_word_embeddings'.format(index_model + 1)
    
    print('\nModel [{}] Training...'.format(1 + index_model))
    
    model = \
    BiLSTM(isa_crf=True,
           words_weights= wb_concatenate,
           pre_trained=True,
           max_len=ner_train.max_len,
           num_words=ner_train.num_words,
           num_tags=ner_train.num_tags,
           learning_rate=LEARNING_RATE,
           dropout=0.5)
    
    All_Historys[file_name_model] = \
    model.fit(ner_train.X_array, 
              ner_train.y_array, 
              batch_size=BATCH_SIZE, 
              epochs=EPOCHS, 
              validation_split=VALIDATION_SPLIT,
              callbacks=[F1Metrics(ner_train.idx2tag), 
                         save_best_model('models/', file_name_model)])
    
    print('-' * 50)
    
    pip_predict_metric(ner_train, 
                       ner_test, 
                       model,
                       path_folder='models/',
                       name_model=file_name_model)
save_stats_to_disk(All_Historys, path_folder='plots/', file_name='8 - Modelo Final + Tweet Embeddings')
%%time

all_f1_score = []
all_recall_score = []
all_precision_score = []

for index_model in range(N_MODELS):

    file_name_model = '{}_model_crf_only_twitter_word_embeddings'.format(index_model + 1)
    
    model = \
    BiLSTM(isa_crf=True,
           words_weights=wb_concatenate,
           pre_trained=True,
           max_len=ner_train.max_len,
           num_words=ner_train.num_words,
           num_tags=ner_train.num_tags,
           learning_rate=LEARNING_RATE,
           dropout=0.5)

    model.load_weights('models/' + file_name_model + '.h5')

    y_pred, y_true = \
    np.argmax(model.predict(ner_test.X_array), axis=-1), \
    np.argmax(ner_test.y_array, -1)
    
    pred_tag, true_tag = \
    ner_train.parser2categorical(y_pred, y_true) 
    
    print('-' * 50)
    print(file_name_model + '[{}] - Metrics...'.format(index_model + 1))
    
    all_precision_score.append(precision_score(pred_tag, true_tag))
    all_recall_score.append(recall_score(pred_tag, true_tag))
    all_f1_score.append(f1_score(pred_tag, true_tag))
    
    all_metrics_fold(pred_tag, true_tag)
print("Average Precision: \t%.2f%%  |  std: (+/- %.2f%%)"%\
      (np.mean(all_precision_score) * 100,\
       np.std(all_precision_score) * 100))

print("Average Recall: \t%.2f%%  |  std: (+/- %.2f%%)"%\
      (np.mean(all_recall_score) * 100,\
       np.std(all_recall_score) * 100))

print("Average F1: \t\t%.2f%%  |  std: (+/- %.2f%%)" %\
      (np.mean(all_f1_score) * 100,\
       np.std(all_f1_score) * 100))
