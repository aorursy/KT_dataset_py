import pandas as pd

from pathlib import Path



input_root_path = Path("../input")
!ls ../input
# read sample_submission.csv

sub = pd.read_csv(input_root_path.joinpath("sample_submission.csv"))

# replace all zero from y_value

all_zeros = sub.copy()

all_zeros["y"] = 0

# save submit

all_zeros.to_csv("baseline_probe_0.0.csv", index=False)

P_bp = -59.28220
idx_1_replace_100 = all_zeros.copy()

idx_1_replace_100["y"][0] = 100

# save submit

idx_1_replace_100.to_csv("probe_0001_100.csv", index=False)

P_1_100 = -59.25187
idx_1_replace_100["y"][0] = 200

# save submit

idx_1_replace_100.to_csv("probe_0001_200.csv", index=False)

P_1_200 = -59.36366
# (2.11)

S_tot = 20000.0 / (2 * P_1_100 - P_bp - P_1_200)

print("Stot is : {:.5f}".format(S_tot))
def calc_y_value_any_idx(P_any_idx, p_bp=P_bp, s_tot=S_tot):

    return (s_tot * (P_any_idx - p_bp) + 10000.0) / 200.0
print("y_1 is : {:.5f}".format(calc_y_value_any_idx(P_1_100)))
idx_2_replace_100 = all_zeros.copy()

idx_2_replace_100["y"][1] = 100

# save submit

idx_2_replace_100.to_csv("probe_0002_100.csv", index=False)

P_2_100 = -59.28220
print("Test_data'rows is {}".format(sub.shape[0]))