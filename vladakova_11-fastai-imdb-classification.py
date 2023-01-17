from fastai.basics import *
from fastai.gen_doc.nbdoc import *
import fastai
from fastai.version import __version__
print(__version__)
from fastai.text import *
imdb = untar_data(URLs.IMDB_SAMPLE)
imdb
!ls -la /root/.fastai/data/imdb_sample
data_lm = (TextList.from_csv(imdb, 'texts.csv', cols='text')
                   .split_by_rand_pct()
                   .label_for_lm()  # Language model does not need labels
                   .databunch())
data_lm.save()
data_lm.show_batch()
# Special tokens
# xxbos: Begining of a sentence
# xxfld: Represent separate parts of a document like title, summary etc., each one will get a separate field and so they will get numbered (e.g. xxfld 1, xxfld 2).
# xxup: If there's something in all caps, it gets lower cased and a token called xxup will get added to it. Words that are fully capitalized, such as “I AM SHOUTING”, are tokenized as “xxup i xxup am xxup shouting“
# xxunk: token used instead of an uncommon word.
# xxmaj: token indicates that there is capitalization of the word. “The” will be tokenized as “xxmaj the“.
# xxrep: token indicates repeated word, if you have 29 ! in a row, (i.e. xxrep 29 !).
data_lm.vocab.itos[:20]
data_lm.train_ds[0][0].data[:10]
learn = language_model_learner(data_lm, AWD_LSTM)
learn.fit_one_cycle(4, 1e-2)
learn.save('mini_train_lm')
learn.save_encoder('mini_train_encoder')
learn.show_results()
learn.predict('When I saw this movie the second time', 100)

learn.predict('I will be ', 100)
data_clas = (TextList.from_csv(imdb, 'texts.csv', cols='text', vocab=data_lm.vocab)
                   .split_from_df(col='is_valid')
                   .label_from_df(cols='label')
                   .databunch(bs=42))
data_clas.show_batch()
learn_cl = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5)
learn_cl.load_encoder('mini_train_encoder')
learn_cl.fit_one_cycle(6, slice(1e-3,1e-2))
learn_cl.save('mini_train_clas')
learn_cl.fit_one_cycle(6, slice(1e-3,1e-2), moms=(0.8, 0.7))
learn_cl.recorder.plot_losses()
learn_cl.show_results()
preds, y, losses = learn_cl.get_preds(with_loss=True)
interp = ClassificationInterpretation(learn_cl, preds, y, losses)
interp.plot_confusion_matrix()
NUM_PREDS = 1000
for i in range(NUM_PREDS):
    reviews.append(learn.predict('', 100))
results = []
for i in range(NUM_PREDS):
    pr = learn_cl.predict(reviews[i])
#     print(pr)
    res = 1 if pr[2][0] < 0.5 else 0
    results.append(res)

from collections import Counter
cnt = Counter(results)
cnt
learn_cl.data.classes
