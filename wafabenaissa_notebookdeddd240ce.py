import pandas



csv1 = pandas.read_csv('../input/clicks_train.csv',low_memory=False)

csv2 = pandas.read_csv('../input/events.csv',low_memory=False)

merged = csv1.merge(csv2, on='display_id')

merged.to_csv("output.csv", index=False)