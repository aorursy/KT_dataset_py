import pandas as pd



other_people_df1 = pd.read_csv("../input/baselinemodeling/submission.csv")



other_people_df2 = pd.read_csv("../input/petfinder-lgbm/submission.csv")



df = other_people_df1[['PetID']]

df['AdoptionSpeed'] = (other_people_df1.AdoptionSpeed + other_people_df2.AdoptionSpeed)/2

df['AdoptionSpeed'] = df.AdoptionSpeed.astype(int)



df.to_csv('submission.csv', index=False)