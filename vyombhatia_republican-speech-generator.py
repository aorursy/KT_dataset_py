# Importing:
from fastai.text.all import *
import os
import shutil
os.makedirs("./new", mode = 0o777, exist_ok = False)
# Defining the list of files:
list  = os.listdir("../input/2020-republican-convention-speeches")

# Copying:
for file in list:
    original = r'../input/2020-republican-convention-speeches/' + file 
    target = r'./new/' + file
    shutil.copyfile(original, target)

# The path where our files are now stored:
path = "./new/"
# Getting those files:
files = get_text_files(path)
# Taking a look at those files:
text = files[7].open().read(); text[:212]
get_elections = partial(get_files)
# Defining the DataLoader finally:
republicans_loader = DataBlock(
    blocks = TextBlock.from_folder(path, is_lm = True),
get_items  = get_elections, splitter = RandomSplitter(0.1)
).dataloaders(path, path=path, bs = 64, seq_len = 70)
# Taking a look at the batches:
republicans_loader.show_batch(max_n = 2)
# Defining the model:
learn = language_model_learner(republicans_loader, AWD_LSTM,
                              drop_mult = 0.3,
                              metrics  = [accuracy, Perplexity()]).to_fp16()
# Training the model for 5 Epochs with the learning rate of 0.001
learn.fine_tune(5, 1e-4)
# Training the model for another 5 Epochs:
learn.fine_tune(5, 1e-2)
learn.predict('I feel that we will win this election',
              50, temperature = 0.75)
# Trying again:
learn.predict('Make america great again',
              60, temperature = 0.8)
# Again:
learn.predict('I love china',
              12, temperature = 0.5)