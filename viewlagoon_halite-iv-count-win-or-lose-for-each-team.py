# Edit this cell



# If you know your team_id, write your team_id. Otherwise None

your_team_id = None



# If you don't know your team_id, write your team_name here.

# Also you can set one of other teams.

# your_team_id has more priority than your_team_name

your_team_name = 'Raine Force'



# Too weak submissions are ignored

# Choose one of these two variables. You should set the other variable to None.

# (When both are specified, score_threshold_from_best_in_team has more priority than number_of_best_agents.)

# I don't know about the most suitable threshold to describe real performance.

score_threshold_from_best_in_team = None

number_of_best_agents = 5



# Ranks of opponent teams

# Both that you are in the range or that you are not in the range, are allowed.

# NOTE: This notebook consumes time which is roughly proportional

#       to the number of episodes of teams within team_rank_range.

#       I don't recommend to set too wide range.

team_rank_range = range(1, 13)

import operator

import pandas as pd

import requests
try:

    episodes_cache

except NameError:

    episodes_cache = {}
LIST_EPISODES = 'https://www.kaggle.com/requests/EpisodeService/ListEpisodes'



def find_team(teams, key, value):

    f = lambda x: x[key] == value

    team = next(filter(f, teams), {})

    return team

    

def request_list_episodes(team_id):

    print(f'request_list_episodes(team_id={team_id}) begin')

    list_episodes = requests.post(LIST_EPISODES, json={'teamId': team_id}).json()

    print(f'request_list_episodes(team_id={team_id}) end')

    return list_episodes
if your_team_id is None:

    RAINE_FORCE_TEAM_ID = 5423724

    list_episodes = request_list_episodes(RAINE_FORCE_TEAM_ID)

    teams = list_episodes['result']['teams']

    your_team_id = find_team(teams, key='teamName', value=your_team_name).get('id', None)

assert your_team_id is not None

list_episodes = request_list_episodes(your_team_id)

episodes_cache[your_team_id] = list_episodes['result']['episodes']

teams = list_episodes['result']['teams']

your_team = find_team(teams, key='id', value=your_team_id)

your_team_name = your_team['teamName']

team_id_to_team_name = {your_team_id: your_team_name}

team_id_to_team_name
def get_target_submission_ids_impl(team_k, submission_id_to_team_id, max_scores, update):

    global score_threshold_from_best_in_team

    global number_of_best_agents

    team_id_k = team_k['id']

    if (not update) and (team_id_k in episodes_cache):

        episodes_k = episodes_cache[team_id_k]

    else:

        episodes_k = request_list_episodes(team_id_k)['result']['episodes']

        # maybe this sort key is wrong to get latest scores

        episodes_k.sort(key=lambda x: x['endTime']['seconds'] + x['endTime']['nanos'] * 1e-9)

        episodes_cache[team_id_k] = episodes_k



    print(f'team_id={team_id_k} team_name={team_k["teamName"]} len(episodes)={len(episodes_k)}')

    submission_id_candidates_k = {}

    for episode_k_e in episodes_k:

        if episode_k_e['type'] != 'public':

            continue  # skip validation

        for agent_k_e_p in episode_k_e['agents']:

            if agent_k_e_p['submission']['teamId'] != team_id_k:

                continue

            submission_id_k_e = agent_k_e_p['submissionId']

            updated_score_k_e = agent_k_e_p['updatedScore']

            if updated_score_k_e is not None:

                submission_id_candidates_k[submission_id_k_e] = updated_score_k_e

    assert 0 < len(submission_id_candidates_k)

    sorted_submission_id_candidates_k = sorted(list(submission_id_candidates_k.items()), key=operator.itemgetter(1, 0), reverse=True)

    assert sorted_submission_id_candidates_k[0][0] == team_k['publicLeaderboardSubmissionId'], f'update required for team_id={team_id_k} team_name={team_k["teamName"]}. Do del episodes_cache[{team_id_k}] and del episodes_cache[your_team_id].'

    max_scores[team_id_k] = sorted_submission_id_candidates_k[0][1]

    count = 0

    for submission_id_k_s, score_k_s in sorted_submission_id_candidates_k:

        s = f'team_id={team_id_k} submission_id={submission_id_k_s} score={score_k_s}'

        if score_threshold_from_best_in_team:

            if score_k_s < max_scores[team_id_k] - score_threshold_from_best_in_team:

                print(f'{s} ignored')

                break

        elif number_of_best_agents <= count:

            print(f'{s} ignored')

            break

        print(f'{s} registered')

        submission_id_to_team_id[submission_id_k_s] = team_id_k

        count += 1

    

def get_target_submission_ids(list_episodes, update=False):

    global episodes_cache

    global your_team

    global team_id_to_team_name



    submission_id_to_team_id = {}  # weak submissions are not registered

    if update:

        episodes_cache = {}

    teams = list_episodes['result']['teams']

    teams.sort(key=lambda x: x['publicLeaderboardRank'] or 1e9)

    max_scores = {}

    for rank in team_rank_range:

        k = rank - 1  # rank is 1-indexed

        team_k = teams[k]

        get_target_submission_ids_impl(team_k, submission_id_to_team_id, max_scores, update)



    if your_team['publicLeaderboardRank'] not in team_rank_range:

        get_target_submission_ids_impl(your_team, submission_id_to_team_id, max_scores, update)



    return submission_id_to_team_id, max_scores
# This method may take a long time if team_rank_range is too wide

submission_id_to_team_id, max_scores = get_target_submission_ids(list_episodes)
match_results = {}

for team_id in max_scores.keys():

    if team_id == your_team_id:

        continue

    team_name = find_team(teams, key='id', value=team_id).get('teamName', str(team_id))

    match_results[team_id] = {'team_name': team_name, 'max_score': round(max_scores[team_id], 1), 'win': 0, 'draw': 0, 'lose': 0}

    

for episode in episodes_cache[your_team_id]:

    episode_data = []

    your_reward = None

    for agent_p in episode['agents']:

        submission_id_p = agent_p['submissionId']

        if submission_id_p in submission_id_to_team_id:

            team_id_p = submission_id_to_team_id[submission_id_p]

            reward = agent_p['reward']

            if reward is None:  # Error

                reward = -1e9

            if team_id_p == your_team_id:

                your_reward = reward

            else:

                episode_data.append((reward, team_id_p))

    if your_reward is None:

        continue

    for reward_p, team_id_p in episode_data:

        assert reward_p is not None, episode_data

        if reward_p < your_reward:

            key = 'win'

        elif reward_p == your_reward:

            key = 'draw'

        else:

            key = 'lose'

        match_results[team_id_p][key] += 1



df = pd.DataFrame(match_results).T

print(your_team_name)

df