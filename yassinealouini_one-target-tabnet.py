import pandas as pd
from pathlib import Path
# The installation procedure is inpsried from this great notebook, thanks for sharing it.
# https://www.kaggle.com/hamishdickson/tabnetmultitaskclassifier



!pip uninstall -y typing # this should avoid  AttributeError: type object 'Callable' has no attribute '_abc_registry'
!cp -r ../input/nfkb-tabnet-model/* . 
!zip model.zip model_params.json network.pt

import sys
sys.path.insert(0, "../input/tabnet-latest")

from pytorch_tabnet.tab_model import TabNetClassifier
BASE_FOLDER = Path("../input/lish-moa/")
TRAIN_FEATURES_PATH = BASE_FOLDER / "train_features.csv"
TEST_FEATURES_PATH = BASE_FOLDER / "test_features.csv"
TRAIN_TARGETS_PATH = BASE_FOLDER / "train_targets_scored.csv"
SAMPLE_SUBMISSION_PATH = BASE_FOLDER / "sample_submission.csv"
MODEL_PATH = "model.zip"
# Category mapping to numbers (similr to what is done in training)
DOSE_MAPPING = {"D1": 0, "D2": 1}
train_targets_df = pd.read_csv(TRAIN_TARGETS_PATH)
train_features_df = pd.read_csv(TRAIN_FEATURES_PATH)
test_features_df = pd.read_csv(TEST_FEATURES_PATH)
test_features_df = pd.read_csv(TEST_FEATURES_PATH)

sample_submission_df = pd.read_csv(SAMPLE_SUBMISSION_PATH)


# Since control is always 0, we can filter those

train_sig_ids = train_features_df.loc[lambda df: df["cp_type"] == "ctl_vehicle", "sig_id"].tolist()

mean_train_targets_dict = train_targets_df.loc[lambda df: ~df["sig_id"].isin(train_sig_ids), :].iloc[:, 1:].mean().to_dict()


for col, mean in mean_train_targets_dict.items():
    sample_submission_df.loc[:, col] = mean


# Predict for one target => nfkb_inhibitor

X_test = test_features_df.loc[lambda df: df["cp_type"] != "ctl_vehicle"].drop(["sig_id", "cp_type"], axis=1)


print(len(X_test))
print(len(test_features_df))

X_test["cp_dose"] = X_test["cp_dose"].map(DOSE_MAPPING)

X_test = X_test.values


model = TabNetClassifier()
model.load_model(MODEL_PATH)
# This is missing
model.preds_mapper = {0: 0, 1: 1}
y_preds = model.predict_proba(X_test)[:, 1]

assert (sum(y_preds) > 0).all()
    
# For the test, if any are from the control group, we set these to 0
test_sig_ids = test_features_df.loc[lambda df: df["cp_type"] == "ctl_vehicle", "sig_id"].tolist()


sample_submission_df.loc[lambda df: ~df["sig_id"].isin(test_sig_ids), "nfkb_inhibitor"] = y_preds

sample_submission_df.loc[lambda df: df["sig_id"].isin(test_sig_ids), :].iloc[:, 1:] = 0
sample_submission_df.std()
assert sample_submission_df["nfkb_inhibitor"].std() > 0
print(sample_submission_df.mean().sort_values())
sample_submission_df.to_csv("submission.csv", index=False)