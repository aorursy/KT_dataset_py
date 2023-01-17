import numpy as np

import pandas as pd

result = pd.read_csv("../input/lish-moa-mpl/submission (60).csv")

result.to_csv("submission.csv", index=False)

result
result = pd.read_csv("submission.csv")

result