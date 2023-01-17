# Importing from FastAI and Pandas:
from fastai.text.all import *
import pandas as pd
# Defining the path where our data is stored:
path = "../input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv"
data = pd.read_csv(path)

# Lets scale down this a little bit since my GPU won't be able to handle data this big:
newdata = data[:2000]
newdata.to_csv("./newdata.csv")

# Taking a look at it:
newdata.head(5)
# Defining the DataBlock:
imdb_clas = DataBlock(
    blocks=(TextBlock.from_df('review', seq_len=72), CategoryBlock),
    get_x=ColReader('text'), get_y=ColReader('sentiment'))
# Further processing it:
imdb = imdb_clas.dataloaders(newdata, bs = 64, is_lm=True)

# Taking a look at it:
imdb.show_batch(max_n=1)
# Let's take the first review:
txt = data['review'].iloc[0]
txt

spacy = WordTokenizer()
toks = first(spacy([txt]))
print(coll_repr(toks, 30))
# Further processing:

tkn = Tokenizer(spacy)
print(coll_repr(tkn(txt), 31))
# Time to fin-tune the model:
learn = text_classifier_learner(imdb,
                               AWD_LSTM,
                               metrics = [accuracy, Perplexity()]).to_fp16()
learn.fit_one_cycle(5, 1e-3)