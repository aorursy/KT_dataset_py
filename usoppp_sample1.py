import pandas as pd

users = pd.read_csv("../input/Users.csv")

submissions=pd.read_csv("../input/Submissions.csv", low_memory=False)
# It's yours to take from here!
len(submissions)