import pandas as pd

import math

import numpy as np



chance_d_estimate_data = pd.read_csv('../input/estimated-chance-democrat-in-2020-by-state/Estimated Chance Democrat in 2020 by State.csv')

turnout_estimate_data = pd.read_csv('../input/projected-2020-voter-turnout-by-state/Projected 2020 Voter Turnout by State.csv')

ec_votes_data = pd.read_csv('../input/electoral-college-votes-by-state/Electoral College Votes.csv')
ec_votes_data['State'] = ec_votes_data['state']

ec_votes_data['EC_Votes'] = ec_votes_data['number of votes']

ec_votes_data = ec_votes_data[['State', 'EC_Votes']]



state_data = chance_d_estimate_data.merge(turnout_estimate_data, on = 'State').merge(ec_votes_data, on = 'State')

state_data['Estimated_Chance_D'] = state_data['Estimated_Chance_D'].apply(lambda chance: chance/100)



state_data
def calculate_entropy(p):

    return (-1 * p * math.log(p, 2)) + (-1 * (1 - p) * math.log(1 - p, 2))



sbd = state_data

sbd['State_Outcome_Bits'] = sbd['Estimated_Chance_D'].apply(lambda chance: calculate_entropy(chance))

sbd['State_Outcome_Bits_Per_Vote'] = sbd.apply(lambda row: row['State_Outcome_Bits'] / row['Turnout'], axis = 1)

sbd['Electoral_College_Bits_Per_Vote'] = sbd.apply(lambda row: row['State_Outcome_Bits_Per_Vote'] * row['EC_Votes'], axis = 1)

state_bit_data = sbd



state_bit_data
def sample_outcome(state_data):

    sample_ec_votes_list = state_data.apply(lambda row: np.random.binomial(1, row['Estimated_Chance_D']) * row['EC_Votes'], axis = 1)

    sample_ec_vote_total = sum(sample_ec_votes_list)

    sample_vote_outcome = int(sample_ec_vote_total >= 269)

    return sample_vote_outcome



mc_sample_size = 1000

outcome_sample_lists = map(lambda x: sample_outcome(state_data), [0] * mc_sample_size)

chance_d_president_estimated = sum(outcome_sample_lists) / mc_sample_size



president_outcome_bits = calculate_entropy(chance_d_president_estimated)

president_outcome_bits_per_electoral_vote = president_outcome_bits / 538



print(chance_d_president_estimated, president_outcome_bits, president_outcome_bits_per_electoral_vote)











sbd = state_bit_data

sbd['Presidential_Outcome_Bits_Square_Per_Vote'] = sbd['Electoral_College_Bits_Per_Vote'].apply(lambda ecbpv: ecbpv * president_outcome_bits_per_electoral_vote)

sbd['Pico_POB2PV'] = sbd['Presidential_Outcome_Bits_Square_Per_Vote'].apply(lambda pob2pv: pob2pv * (10**12))

sbd = sbd[['State', 'Pico_POB2PV']]

sbd = sbd.sort_values(by = ['Pico_POB2PV'], ascending = False)

sbd['OB_Rank'] = range(1, sbd.shape[0] + 1)

state_bit_data = sbd



state_bit_data