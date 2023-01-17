import pandas as pd

from pathlib import Path
%ls ../input/bengaliai-cv19/
%ls ../input/bengali-handwritten-classification-submits/
dataset_path = Path("../input/bengaliai-cv19")

local_submit_path = Path("../input/bengali-handwritten-classification-submits")

submit_path = dataset_path / "sample_submission.csv"

local_path = local_submit_path / "ResNet-18_submission.csv"
submission = pd.read_csv(submit_path)
submission.head()
local_submit = pd.read_csv(local_path)
local_submit.head()
target_col = "target"

id_col = "row_id"

for i, row in local_submit.iterrows():

    submission.loc[submission[id_col] == row[id_col],target_col] = row[target_col]
submission
submission.to_csv("submission.csv", index=False)