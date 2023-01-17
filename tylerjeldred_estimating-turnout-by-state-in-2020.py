import pandas as pd

from sklearn import linear_model

import math



turnout_rates_historical_data = pd.read_csv('../input/turnout-rates-by-state-ballotpedia/Turnout Rates by State (Ballotpedia).csv')

vep_2020_data = pd.read_csv('../input/voting-eligible-population-2020-us-census/Voting Eligible Population 2020 by State (US Census).csv')

vep2 = vep_2020_data

vep2["Voting_Eligible_Population"] = vep2["votingEligiblePop"]

vep_2020_data = vep2



vep_2020_data
trhd = turnout_rates_historical_data

trhd['State'] = trhd['State'].astype('category')

trhd = trhd.melt(id_vars = ['State'])

trhd['Year'] = trhd['variable'].apply(lambda x: int(x))

trhd['Turnout_Rate'] = trhd['value']

turnout_rates_historical_states = trhd['State']

trhd = trhd.dropna()

trhd['Presidential_Election'] = trhd['Year'].apply(lambda x: x % 4 == 0)

states = turnout_rates_historical_states.cat.categories.tolist()

trhd_dummies = pd.get_dummies(trhd['State'])

trhd = pd.concat([trhd[['Year', 'Presidential_Election', 'Turnout_Rate']], trhd_dummies], axis=1, ignore_index=True)

trhd.columns = ['Year', 'Presidential_Election', 'Turnout_Rate'] + states

trhd['Turnout_Rate'] = trhd['Turnout_Rate'].apply(lambda x: float(x.strip("%")))

turnout_rates_historical_data = trhd



turnout_rates_historical_data
turnout_rates_historical_features = turnout_rates_historical_data.loc[:, turnout_rates_historical_data.columns != 'Turnout_Rate']

turnout_rates_historical_responses = turnout_rates_historical_data[['Turnout_Rate']]



regressor = linear_model.LinearRegression()

regressor.fit(turnout_rates_historical_features, turnout_rates_historical_responses)
trp = pd.DataFrame(data = { 'State': states, 'Year': [2020] * len(states), 'Presidential_Election': [True] * len(states) })

trp_dummies = pd.get_dummies(trp['State'])

trp = pd.concat([trp[['Year', 'Presidential_Election']], trp_dummies], axis=1, ignore_index=True)

trp.columns = ['Year', 'Presidential_Election'] + states

trp['Turnout_Rate'] = regressor.predict(trp)



trp = trp.drop('Presidential_Election', axis = 1)

trp['State'] = trp.apply(lambda row: trp.columns[list(row).index(1)], axis = 1)

trp = trp[['State', 'Year', 'Turnout_Rate']]

turnout_rates_projected = trp



turnout_rates_projected

turnout_rates_projected = turnout_rates_projected.merge(vep_2020_data, on = 'State')

turnout_rates_projected["Turnout"] = turnout_rates_projected.apply(lambda row: int(round(row['Turnout_Rate'] * row['Voting_Eligible_Population'] * (1/100))), axis = 1)

turnout_rates_projected = turnout_rates_projected[['State', 'Year', 'Turnout_Rate', 'Voting_Eligible_Population', 'Turnout']]



turnout_rates_projected
turnout_rates_projected.to_csv('./Projected 2020 Voter Turnout by State.csv', index = False)