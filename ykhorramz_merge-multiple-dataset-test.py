import pandas as pd

scientists_df = pd.read_csv('../input/scientist-migrations/ORCID_migrations_2016_12_16_by_person.csv', index_col = 0)

country_df = pd.read_csv('../input/world-development-indicators/Country.csv')

country_df.head(2)