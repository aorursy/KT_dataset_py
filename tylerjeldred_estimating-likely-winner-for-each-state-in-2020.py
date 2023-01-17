import pandas as pd

import numpy as np



scorecard_2016_data = pd.read_csv('../input/2016-electoral-scorecard-cook/2016 Electoral Scorecard (Cook).csv')

scorecard_2020_data = pd.read_csv('../input/2020-electoral-scorecard-cook/2020 Electoral Scorecard (Cook).csv')

chance_2016_data = pd.read_csv('../input/chance-democratic-by-state-2016-fivethirtyeight/Chance Democratic by State 2016 (FiveThirtyEight).csv')



scorecard_chance_2016_data = scorecard_2016_data.merge(chance_2016_data, on = 'State')



combined_states_list = scorecard_chance_2016_data['State']



scorecard_only_states_list = np.setdiff1d(scorecard_chance_2016_data['State'], combined_states_list)

chance_only_states_list = np.setdiff1d(chance_2016_data['State'], combined_states_list)



scorecard_chance_2016_data
scad = scorecard_chance_2016_data[['Rating', 'Chance D']].groupby(['Rating']).mean()

scad['Chance D'] = scad['Chance D'].apply(lambda x: round(x, 1))

scorecard_chance_aggregate_data = scad



scorecard_chance_aggregate_data
sce2d = scorecard_2020_data.merge(scorecard_chance_aggregate_data, on = 'Rating')

sce2d['Estimated_Chance_D'] = sce2d['Chance D']

sce2d = sce2d[['State', 'Estimated_Chance_D']]

scorecard_chance_estimate_2020_data = sce2d



scorecard_chance_estimate_2020_data
scorecard_chance_estimate_2020_data.to_csv('./Estimated Chance Democrat in 2020 by State.csv', index = False)