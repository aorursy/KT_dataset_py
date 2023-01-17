#Precisamos fazer o download do ambiente de Connect X:
# Vamos precisar de um kaggle-enviroments customizado para a avaliação.
!pip install git+https://github.com/matheusgmaia/kaggle-environments
    
#Criar ambiente
from kaggle_environments import evaluate, make, utils

env = make("connectx", debug=True)
env.render()
def init_global_variables(obs):
    global GLOBAL_VARIABLES_ARE_SET
    global MY_AGENT
    global MOVES
    
    GLOBAL_VARIABLES_ARE_SET = True
    MY_AGENT = obs['mark']
    if MY_AGENT == 2:
        MOVES += 1
    obs['opponent'] = 1 if MY_AGENT == 2 else 2

def make_board(obs, cfg, agent):
    ''' Returns an integer representing the board for the positions of a given agent and another integer for all occupied positions in the board '''
    indices_to_add_zero = []
    for row in range(cfg['rows']):
        indices_to_add_zero.append(row * cfg['columns'] + cfg['rows'] + row + 1) # Indices next to the end of each row

    agent_positions = ['1' if p == agent else '0' for p in obs['board']] # List with '1's where agent has a piece
    occupied_positions = ['1' if p != 0 else '0' for p in obs['board']] # List with '1's where there's a piece
    
    for index in indices_to_add_zero: # Adds a '0' in the end of each row so that when we check connections false-positives don't happen
        agent_positions.insert(index, '0')
        occupied_positions.insert(index, '0')

    agent_positions_str = ''.join(agent_positions)
    agent_positions_int = int(agent_positions_str, 2)
    
    occupied_positions_str = ''.join(occupied_positions)
    occupied_positions_int = int(occupied_positions_str, 2)
    
    return agent_positions_int, occupied_positions_int

ONES_COLUMN = int(('0' * 7 + '1') * 6, 2)
def make_move2(player_positions, occupied_positions, column):
    ''' Makes a move for player_positions on the given column and returns the new player_positions and the new occupied board  '''
    opponent_positions = player_positions ^ occupied_positions
    new_occupied_positions = occupied_positions | ((occupied_positions & (ONES_COLUMN << (7 - column))) << 8) + (1 << (7 - column))
    new_player_positions = opponent_positions ^ new_occupied_positions
    
    global MOVES
    MOVES += 1
    return new_player_positions, new_occupied_positions

def make_move(occupied_positions, column):
    ''' Adds a piece on the given column on occupied_positions and then returns it '''
    new_occupied_positions = occupied_positions | ((occupied_positions & (ONES_COLUMN << (7 - column))) << 8) + (1 << (7 - column))
    return new_occupied_positions

def count_ones(number):
    ''' Returns the number of 1s in the representation of a given integer as a binary '''
    number_str = str(bin(number))
    return number_str.count('1')

def bit_not(n, numbits=7):
    ''' Performs the not operation bit by bit on a given number. We use 7 as default for numbits since it is the number of columns '''
    return (1 << numbits) - 1 - n

def get_available_columns(occupied_positions):
    ''' Returns a list with '1' if the column still has space and '0' otherwise '''
    top_row = occupied_positions >> 41 # num_rows * (num_columns + 1) + 1 = 41
    top_row = format(top_row, '07b')
    available_columns = [c for c in range(len(top_row)) if top_row[c] == '0']
    
    return available_columns
m = '00000100' * 6
ac = get_available_columns(int(m, 2))
print(ac)
OFFSET = {
    'row': 1,
    'column': 8,
    'upper_diagonal': 7,
    'lower_diagonal': 9
}
WEIGHTS = {
    'win': 100000,
    'triples': {
        'row': 1500,
        'column': 2000,
        'upper_diagonal': 1000,
        'lower_diagonal': 1000,
        'surrounded': -5000
    },
    'column': [50, 150, 250, 500, 250, 150, 50]
}
def count_quartets(player_positions, direction):
    ''' Returns the number of quartets from the same player in a specific direction '''
    offset = OFFSET[direction]
    mask = player_positions & (player_positions >> offset)
    if mask & (mask >> (offset * 2)):
        return 1        
    return 0

def eval_wins(player_positions):
    ''' Returns a high number if there is a quartet for the given player_positions '''
    result = 0
    result += count_quartets(player_positions, 'row')
    result += count_quartets(player_positions, 'column')
    result += count_quartets(player_positions, 'upper_diagonal')
    result += count_quartets(player_positions, 'lower_diagonal')
    if result:
        return WEIGHTS['win']
    return 0
def make_mask(player_positions, direction):
    ''' Returns a number that represents the number of triples in a specific direction '''
    offset = OFFSET[direction]
    # Example: player_positions = 0011100
    mask = player_positions & (player_positions >> offset) # mask = 0011100 & 0001110 = 0001100
    mask = mask & (mask << offset) # mask = 0001100 & 0011000 = 0001000
    return mask

def count_surrounded_triples(mask, occupied_positions, direction):
    ''' Returns the number of triples surrounded by an oponent piece in a specific direction '''
    offset = OFFSET[direction]
    # Example: occupied_positions = 0111100 <- Opponent piece on the left of player_positions
    other_mask = occupied_positions & (occupied_positions >> offset) # other_mask = 0111100 & 0011110 = 0011100
    other_mask = other_mask & (other_mask << offset) # other_mask = 0011100 & 0111000 = 0011000
    other_mask = other_mask | mask # other_mask = 0011000 | 0001000 = 0011000
    other_mask = other_mask & (other_mask >> offset) # other_mask = 0011000 & 0001100 = 0001000
    
    return count_ones(other_mask)

def eval_triples_on_direction(player_positions, occupied_positions, direction):
    mask = make_mask(player_positions, direction)
    triples = count_ones(mask)
    # result = triples * WEIGHTS['triples'][direction]
    if triples:
        surrounded_triples = count_surrounded_triples(mask, occupied_positions, direction)
        if surrounded_triples / 2 >= triples:
            triples = 0
        #result += surrounded_triples * WEIGHTS['triples']['surrounded']
    
    return triples * WEIGHTS['triples'][direction]
    
def eval_all_triples(player_positions, occupied_positions):
    result = 0
    result += eval_triples_on_direction(player_positions, occupied_positions, 'row')
    result += eval_triples_on_direction(player_positions, occupied_positions, 'column')
    result += eval_triples_on_direction(player_positions, occupied_positions, 'upper_diagonal')
    result += eval_triples_on_direction(player_positions, occupied_positions, 'lower_diagonal')
    
    return result
def eval_positions(player_positions, occupied_positions):
    result = 0
    result += eval_wins(player_positions) + eval_all_triples(player_positions, occupied_positions)
    opponent_positions = player_positions ^ occupied_positions
    result -= (eval_wins(player_positions) + eval_all_triples(opponent_positions, occupied_positions))
    #print('result: ', result)
    
    return result
def minimax(player_positions, occupied_positions, depth, maximizing_player):
    global MOVES
    score = 0
    best_column = None
    
    opponent_positions = player_positions ^ occupied_positions
    win_result = eval_wins(player_positions) - eval_wins(opponent_positions)
    if depth == 0 or win_result != 0 or MOVES > 41:
        if win_result:
            score = win_result
        else:
            score = eval_positions(player_positions, occupied_positions)
    else:
        available_columns = get_available_columns(occupied_positions)
        if maximizing_player:
            score = -float('inf')
            for column in available_columns:
                new_occupied_positions = make_move(occupied_positions, column)
                MOVES += 1
                new_player_positions = opponent_positions ^ new_occupied_positions
                new_score = minimax(new_player_positions, new_occupied_positions, depth - 1, False)[0] + WEIGHTS['column'][column]
                #print(column, ': ', new_score)
                MOVES -= 1
                if new_score > score:
                    score = new_score
                    best_column = column
        else: # Minimizing player
            score = float('inf')
            for column in available_columns:
                new_occupied_positions = make_move(occupied_positions, column)
                MOVES += 1
                new_score = minimax(player_positions, new_occupied_positions, depth - 1, True)[0] + WEIGHTS['column'][column]
                MOVES -= 1
                if new_score < score:
                    score = new_score
                    best_column = column
    return score, best_column
    
#Sua vez
MOVES = 0 # number of moves made so far
PREVIOUS_BOARD = None
MY_AGENT = None
GLOBAL_VARIABLES_ARE_SET = False
def my_agent(obs, cfg):
    global MOVES
    if not GLOBAL_VARIABLES_ARE_SET:
        init_global_variables(obs)
    agent_positions, occupied_positions = make_board(obs, cfg, obs['mark'])
    column = minimax(agent_positions, occupied_positions, 4, True)[1]
    MOVES += 1

    return column
env.reset()
env.run(["greedy", my_agent]) #Agente definido em my_agent versus angente randômico.
env.render(mode="ipython", width=500, height=450)
# Play as first position against random agent.
trainer = env.train([None, "greedy"])

observation = trainer.reset()

while not env.done:
    my_action = my_agent(observation, env.configuration)
    print("Ação do seu agente: Coluna", my_action+1)
    observation, reward, done, info = trainer.step(my_action)
    env.render(mode="ipython", width=100, height=90, header=False, controls=False)
env.render()
# "None" represents which agent you'll manually play as (first or second player).
env.play([my_agent, None], width=500, height=450) #Altere "rules" por my_agent para jogar contra o seu agente
def mean_win_draw(rewards):
    return sum( 1 for r in rewards if (r[0] == 1 or r[0] == 0.)) / len(rewards)

# Run multiple episodes to estimate its performance.
vs_random = mean_win_draw(evaluate("connectx", [my_agent, "random"], num_episodes=10))
print("My Agent vs Random Agent:", vs_random)

vs_negamax = mean_win_draw(evaluate("connectx", [my_agent, "negamax"], num_episodes=10))
print("My Agent vs Negamax Agent:", vs_negamax)

vs_rules = mean_win_draw(evaluate("connectx", [my_agent, "rules"], num_episodes=10))
print("My Agent vs Rule Agent:", vs_rules)

vs_greedy = mean_win_draw(evaluate("connectx", [my_agent, "greedy"], num_episodes=10))
print("My Agent vs Greedy Agent:", vs_greedy)

vs_random = mean_win_draw(evaluate("connectx", ["random", my_agent], num_episodes=10))
print("My Agent vs Random Agent:", vs_random)

vs_negamax = mean_win_draw(evaluate("connectx", ["negamax", my_agent], num_episodes=10))
print("My Agent vs Negamax Agent:", vs_negamax)

vs_rules = mean_win_draw(evaluate("connectx", ["rules", my_agent], num_episodes=10))
print("My Agent vs Rule Agent:", vs_rules)

vs_greedy = mean_win_draw(evaluate("connectx", ["greedy", my_agent], num_episodes=10))
print("My Agent vs Greedy Agent:", vs_greedy)
import csv

seu_nome = "VALMIR_JR"

rows = [['Id', 'Predicted'],['random',vs_random],[ 'negamax', vs_negamax],[ 'rules', vs_rules],[ 'greedy', vs_greedy]]
f = open(seu_nome+'-ConnectX.csv', 'w')
with f:
    writer = csv.writer(f)
    for row in rows:
        writer.writerow(row)
import inspect
import os

def write_agent_to_file(function, file):
    with open(file, "a" if os.path.exists(file) else "w") as f:
        f.write(inspect.getsource(function))
        print(function, "written to", file)

write_agent_to_file(my_agent, "submission.py")
# Note: Stdout replacement is a temporary workaround.
import sys
out = sys.stdout
submission = utils.read_file("/kaggle/working/submission.py")
agent = utils.get_last_callable(submission)
sys.stdout = out

env = make("connectx", debug=True)
env.run([agent, agent])
print("Success!" if env.state[0].status == env.state[1].status == "DONE" else "Failed...")