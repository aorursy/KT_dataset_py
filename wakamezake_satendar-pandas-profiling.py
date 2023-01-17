import numpy as np

import pandas as pd

import pandas_profiling as pdp

import os

import warnings

warnings.filterwarnings("ignore")

from pathlib import Path

from IPython.display import HTML
root = Path("../input")

train = pd.read_csv(root.joinpath("train.csv"))

profile = pdp.ProfileReport(train)

profile.to_file(outputfile="train.html")

HTML(filename='train.html')
test = pd.read_csv(root.joinpath("test.csv"))

profile = pdp.ProfileReport(test)

profile.to_file(outputfile="test.html")

HTML(filename='test.html')