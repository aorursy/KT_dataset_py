import numpy as np
import pandas as pd
import pandas_profiling as pdp
train = pd.read_csv("../input/train_V2.csv")
# test = pd.read_csv("../input/test_V2.csv")
profile = pdp.ProfileReport(train.drop(columns=["Id", "groupId", "matchId"]))
profile.to_file(outputfile="train_outputfile.html")
from IPython.display import HTML
HTML(filename="train_outputfile.html")