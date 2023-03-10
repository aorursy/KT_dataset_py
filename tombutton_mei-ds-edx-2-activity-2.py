import pandas as pd

heathrow_2015_data = pd.read_csv("../input/edexcelldsheathrow/heathrow-2015.csv")
heathrow_2015_data.head(6)

heathrow_2015_data.dtypes
heathrow_2015_data['Daily Mean Temperature'].describe()
heathrow_2015_data = pd.read_csv("../input/edexcelldsheathrow/heathrow-2015.csv")
heathrow_2015_data['Daily Mean Temperature'].describe()
heathrow_2015_data['Daily Total Rainfall'].describe()
heathrow_2015_data['Daily Total Rainfall'] = heathrow_2015_data['Daily Total Rainfall'].replace({'tr': 0})
heathrow_2015_data['Daily Total Rainfall'] = heathrow_2015_data['Daily Total Rainfall'].astype('float')
heathrow_2015_data['Daily Total Rainfall'].describe()
heathrow_1987_data['Daily Total Rainfall'] = heathrow_1987_data['Daily Total Rainfall'].replace({'tr': 0.025})
heathrow_1987_data['Daily Total Rainfall'] = heathrow_1987_data['Daily Total Rainfall'].astype('float')
heathrow_1987_data['Daily Total Rainfall'].describe()
heathrow_2015_data['Mean Cardinal Direction'].value_counts()
heathrow_1987_data['Mean Cardinal Direction'].value_counts()
heathrow_2015_data['Mean Cardinal Direction'].value_counts()
heathrow_2015_data['Mean Cardinal Direction'].value_counts()
heathrow_2015_data = pd.read_csv("../input/edexcelldsheathrow/heathrow-2015.csv")
print(heathrow_2015_data['Daily Mean Temperature'].describe())
print(heathrow_2015_data['Mean Cardinal Direction'].value_counts())
heathrow_1987_data = pd.read_csv("../input/edexcelldsheathrow/heathrow-1987.csv")
print(heathrow_1987_data['Daily Mean Temperature'].describe())
print(heathrow_1987_data['Mean Cardinal Direction'].value_counts())
