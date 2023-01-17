from pathlib import Path

from fastai.text import *

import pandas as pd
csv_path = Path('/kaggle/input/twitter-airline-sentiment')

path = '/kaggle/output'

df = pd.read_csv(csv_path / 'Tweets.csv')

df.head()
import re



strip_handles = lambda text: re.sub(r'@[^\s]+\s', '', text)

remove_urls = lambda text: re.sub(r'\shttps?:[^\s]+\s?', '', text)



tweet = df.at[42, 'text']

tweet, strip_handles(tweet), remove_urls('visit this site http://www.google.com')
class StripWeirdThings(PreProcessor):

    def process_one(self, item: str) -> str:

        return remove_urls(strip_handles(item))

    

processor = [StripWeirdThings(), TokenizeProcessor(), NumericalizeProcessor()]
data_lm = (TextList.from_df(df, path, cols=['text'], 

            processor=processor)

           .split_by_rand_pct(0.2, seed=42)

           .label_for_lm()

           .databunch(bs=42, num_workers=1))

data_lm.show_batch()
learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.3)

learn.lr_find()

learn.recorder.plot(suggestion=True)

min_grad_lr = learn.recorder.min_grad_lr

min_grad_lr
learn.fit_one_cycle(10, 1e-02, moms=(0.8, 0.7))

learn.recorder.plot_losses()

learn.save('lm_head.model')

learn.freeze_to(-2)

learn.fit_one_cycle(10, 1e-3, moms=(0.9, 0.8))

learn.recorder.plot_losses()
learn.save('lm_head_2.model')
learn.show_results()
learn.save('lm_head_2.model')

learn.save_encoder('lm_encoder')
def create_label(sent_and_reason):

    sent = sent_and_reason[0]

    reason = sent_and_reason[1]

    if sent == 'negative':

        return reason

    else:

        return sent



df['label'] = df[['airline_sentiment', 'negativereason']].apply(create_label, axis=1)

df['label'].unique()
data_clas = (TextList.from_df(df, path, cols='text', vocab=data_lm.vocab, processor=processor)

             .split_by_rand_pct(0.2, seed=42)

             .label_from_df(cols='label')

             .databunch(bs=64))



data_clas.save('data_clas_export.pkl')
data_clas.show_batch()
learn = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.3)

learn.load_encoder('lm_encoder')

learn.lr_find()

learn.recorder.plot(suggestion=True)

min_grad_lr = learn.recorder.min_grad_lr
learn.fit_one_cycle(6, 1e-02)

learn.recorder.plot_losses()
learn.freeze_to(-2)

learn.fit_one_cycle(4, slice(5e-3, 2e-3), moms=(0.8,0.7))

learn.recorder.plot_losses()
learn.unfreeze()

learn.fit_one_cycle(4, slice(2e-3/100, 2e-3), moms=(0.8,0.7))
learn.recorder.plot_losses()
def predict(learn, tweet):

    learn.freeze()

    learn = learn.to_fp32()

    interp = TextClassificationInterpretation.from_learner(learn)

    interp.show_intrinsic_attention(tweet)

    return learn.predict(tweet)
predict(learn, df.at[8992, 'text'])
predict(learn, df.at[1000, 'text'])