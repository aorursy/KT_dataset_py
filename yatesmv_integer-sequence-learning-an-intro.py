import pandas as pd 
import matplotlib as mpl
from matplotlib import pyplot as plt

#read in files
test = pd.read_csv("../input/test.csv")
train = pd.read_csv("../input/train.csv")
seqs = {ix: pd.Series(x['Sequence'].split(',')) for ix, x in train.iterrows()}