from learntools.core import binder

binder.bind(globals())

from learntools.game_ai.ex3 import *
q_1.hint()
# Check your answer (Run this code cell to receive credit!)

q_1.solution()
# Fill in the blank

num_leaves = 7*7*7



# Check your answer

q_2.check()
# Lines below will give you a hint or solution code

#q_2.hint()

#q_2.solution()
# Fill in the blank

selected_move = 3



# Check your answer

q_3.check()
# Lines below will give you a hint or solution code

#q_3.hint()

#q_3.solution()
#q_4.hint()
# Check your answer (Run this code cell to receive credit!)

q_4.solution()
def my_agent(obs, config):

    N_STEPS = 3

    import random

    import numpy as np

#--------------------

    def drop_piece(grid, col, mark, config): # returns the grid appearing if piece of type mark has been dropped at column col

        next_grid = grid.copy()

        for row in range(config.rows-1, -1, -1):

            if next_grid[row][col] == 0:

                break

        next_grid[row][col] = mark

        return next_grid

    

# ------------ #

    def check_window(window, num_discs, piece, config): # checks if window satisfies the heuristic conditions (will use it with num_discs={3,4} and piece={mark, mark%2+1})

        return (window.count(piece) == num_discs and window.count(0) == config.inarow-num_discs)

# ------------ #

    def count_windows(grid, num_discs, piece, config): # counts how many windows have num_discs discs of type piece in a row (along all dirs) in grid named grid

        num_windows = 0

    # horizontal

        for row in range(config.rows):

            for col in range(config.columns-(config.inarow-1)):

                window = list(grid[row, col:col+config.inarow])

                if check_window(window, num_discs, piece, config):

                    num_windows += 1

    # vertical

        for row in range(config.rows-(config.inarow-1)):

            for col in range(config.columns):

                window = list(grid[row:row+config.inarow, col])

                if check_window(window, num_discs, piece, config):

                    num_windows += 1

    # positive diagonal

        for row in range(config.rows-(config.inarow-1)):

            for col in range(config.columns-(config.inarow-1)):

                window = list(grid[range(row, row+config.inarow), range(col, col+config.inarow)])

                if check_window(window, num_discs, piece, config):

                    num_windows += 1

    # negative diagonal

        for row in range(config.inarow-1, config.rows):

            for col in range(config.columns-(config.inarow-1)):

                window = list(grid[range(row, row-config.inarow, -1), range(col, col+config.inarow)])

                if check_window(window, num_discs, piece, config):

                    num_windows += 1

        return num_windows



# ---- HELPER FUNCTIONS FOR MINIMAX ---- #    

    def get_heuristic(grid, mark, config): # returns score 

        num_threes = count_windows(grid, 3, mark, config) # A

        num_threes_opp = count_windows(grid, 3, mark%2+1, config) # B

        num_fours_opp = count_windows(grid, 4, mark%2+1, config) # C

        num_fours = count_windows(grid, 4, mark, config) # D

        A = 10

        B = 100

        C = 1e08

        D = 1e07

        score = A*num_threes - B*num_threes_opp - C*num_fours_opp + D*num_fours

        return score

    

# ----------- #

    def is_terminal_window(window, config): # Helper function for minimax: checks if agent or opponent has four in a row in the window

        return window.count(1) == config.inarow or window.count(2) == config.inarow   

    

# ------------ #

    def is_terminal_node(grid, config): # Helper function for minimax: checks if game has ended

        # Check for draw 

        if list(grid[0, :]).count(0) == 0: # the number of empty circles in the very top row, along all columns, is 0

            return True

        

        # Check for win: horizontal, vertical, or diagonal

        # horizontal 

        for row in range(config.rows):

            for col in range(config.columns-(config.inarow-1)):

                window = list(grid[row, col:col+config.inarow])

                if is_terminal_window(window, config):

                    return True

        # vertical

        for row in range(config.rows-(config.inarow-1)):

            for col in range(config.columns):

                window = list(grid[row:row+config.inarow, col])

                if is_terminal_window(window, config):

                    return True

        # positive diagonal

        for row in range(config.rows-(config.inarow-1)):

            for col in range(config.columns-(config.inarow-1)):

                window = list(grid[range(row, row+config.inarow), range(col, col+config.inarow)])

                if is_terminal_window(window, config):

                    return True

        # negative diagonal

        for row in range(config.inarow-1, config.rows):

            for col in range(config.columns-(config.inarow-1)):

                window = list(grid[range(row, row-config.inarow, -1), range(col, col+config.inarow)])

                if is_terminal_window(window, config):

                    return True

        return False

    

# --- MINIMAX function --- #

    #def minimax(node, depth, maximizingPlayer, mark, config):

    #    is_terminal = is_terminal_node(node, config) # checks if current node/timestep results is a win/draw

    #    valid_moves = [c for c in range(config.columns) if node[0][c] == 0] # it is valid if the very top row has empty cirlce at that column c

    #    

    #    if depth == 0 or is_terminal: # if no more depth to explore or current node is a win/draw

    #        return get_heuristic(node, mark, config) # returns score of current table configuration

    #    

    #    if maximizingPlayer: # if agent is to play?

    #        value = - np.Inf # agent wants to increase score as much as possible

    #        for col in valid_moves: # for each of the possible valid moves at this timestep

    #            child = drop_piece(node, col, mark, config) # what appears after dropping piece of type mark in column col (when grid=node, i.e. now)

    #            value = max(value, minimax(child, depth-1, False, mark, config)) # update value (this initially starting from -Inf)

                # minimax(child, depth-1,...) is the next step to iteratively calculate the maximum obtainable value after putting piece in column col

    #        return value # returns maximum value obtained by searching through all possible valid moves at this timestep.

    #    else: # if opponent is to play?

    #        value = np.Inf # opponent wants to decrease value as much as possible

    #        for col in valid_moves:

    #            child = drop_piece(node, col, mark%2+1, config) # resulting table after dropping piece of type mark%2+1 in column col (when grid=node, i.e. now)

    #            value = min(value, minimax(child, depth-1, True, mark, config))

    #        return value

    

    def alphabeta(node, depth, alpha, beta, maximizingPlayer, mark, config):

        is_terminal = is_terminal_node(node, config)

        valid_moves = [c for c in range(config.columns) if node[0][c]==0]

        

        if depth==0 or is_terminal:

            return get_heuristic(node, mark, config)

        

        if maximizingPlayer:

            value = - np.Inf

            for col in valid_moves:

                child = drop_piece(node, col, mark, config)

                value = max(value, alphabeta(child, depth-1, alpha, beta, False, mark, config))

                alpha = max(alpha, value)

                if alpha >= beta: # beta cut-off

                    break

            return value

        else:

            value = np.Inf

            for col in valid_moves:

                child = drop_piece(node, col, mark, config)

                value = min(value, alphabeta(child, depth-1, alpha, beta, True, mark, config))

                beta = min(beta, value)

                if beta <= alpha:

                    break

            return value

            

    # -------    

    def score_move(grid, col, mark, config, nsteps):

            next_grid = drop_piece(grid, col, mark, config)

            score = alphabeta(next_grid, nsteps-1, -np.Inf, np.Inf, True, mark, config)

            return score

    

    

    

# --- main() part of the program --- #

    valid_moves = [col for col in range(config.columns) if obs.board[col]==0]

    grid = np.asarray(obs.board).reshape(config.rows, config.columns)

    

    scores = dict(zip(valid_moves, [score_move(grid, col, obs.mark, config, N_STEPS) for col in valid_moves]))

    scoremaximizing_cols = [key for key in scores.keys() if scores[key]==max(scores.values())]

    

    return random.choice(scoremaximizing_cols)
# Run this code cell to get credit for creating an agent

q_5.check()
import inspect

import os



def write_agent_to_file(function, file):

    with open(file, "a" if os.path.exists(file) else "w") as f:

        f.write(inspect.getsource(function))

        print(function, "written to", file)



write_agent_to_file(my_agent, "submission.py")
from kaggle_environments import make, evaluate, utils, agent

import sys



out = sys.stdout

submission = utils.read_file("/kaggle/working/submission.py")

agent = agent.get_last_callable(submission) # careful because the tutorial's way of using utils.get_last_callable(submission) doesn't work.

sys.stdout = out # need to replace utils with agent (i.e. agent.get_last_callable(submission))



env = make("connectx", debug=True)

env.run([agent, agent])

print("Success!" if env.state[0].status == env.state[1].status == "DONE" else "Failed...")