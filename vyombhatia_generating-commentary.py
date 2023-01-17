import os
import pandas as pd
from fastai.text.all import *
# Defining where it stored:
path = "../input/can-generate-automatic-commentary-for-ipl-cricket/IPL_Match_Highlights_Commentary.csv"

# Taking a look at it:
data = pd.read_csv(path)
data.head(3)
os.makedirs("./input", exist_ok = False)
# Taking a look at the shape of the csv
data.shape
for i in range(10790):
    commentary = data.Commentary[i]               # This is the text of row 'i'.
    f = open("./input/" + str(i) + ".txt", "w")   # This writes a new text file.
    f.write(commentary)                             
    f.close()                                     # This saves it.
f = open("./input/0.txt", "r")
print(f.read())
# Grabbing the text files now:

inputpath = "./input"
textfiles = get_text_files(inputpath)
get_comments = partial(get_files)
commentary_loader = DataBlock(
    blocks = TextBlock.from_folder(inputpath, is_lm = True),
    get_items = get_comments, splitter = RandomSplitter(0.1)
).dataloaders(inputpath, path = inputpath, bs = 64, seq_len = 70)
commentary_loader.show_batch(max_n = 2)
learn = language_model_learner(commentary_loader, AWD_LSTM, drop_mult = 0.3, metrics = accuracy).to_fp16()
learn.fit_one_cycle(5, 2e-2)
learn.fit_one_cycle(5, 1e-2)
learn.predict("and there goes the ball towards", 20, temperature = 0.5)
learn.predict("the ball flies", 20, temperature = 0.7)
learn.predict("Gayle hits", 20, temperature = 0.89)