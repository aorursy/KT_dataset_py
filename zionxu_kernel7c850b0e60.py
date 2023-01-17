import numpy as np

import pandas as pd 

import os



sub = pd.read_csv("../input/submissions/submission.csv")

sub.to_csv("submission.csv", index=False)