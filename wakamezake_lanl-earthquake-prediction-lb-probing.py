import pandas as pd

from pathlib import Path
%ls ../input/LANL-Earthquake-Prediction/
dataset_path = Path("../input/LANL-Earthquake-Prediction/")

submit_path = dataset_path / "sample_submission.csv"
# read sample_submission.csv

sub = pd.read_csv(submit_path)

sub.head()
target_col = "time_to_failure"

all_zeros = sub.copy()

all_zeros[target_col] = 0

# save submit

all_zeros.to_csv("baseline_probe_0.0.csv", index=False)
for score in [10, 20, 30]:

    sub = sub.copy()

    sub[target_col] = score

    # save submit

    sub.to_csv("baseline_probe_{}.csv".format(score), index=False)