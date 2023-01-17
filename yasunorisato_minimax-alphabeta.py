# v21: next_grid & N_STEP=2 & centered (July 16, 2020)
# Minimax_alphabeta (Exercise: N-Step Lookahead)
# Version control (https://www.kaggle.com/ajeffries/connectx-getting-started)



# 1. Enable Internet in the Kernel (Settings side pane)



# 2. Curl cache may need purged if v0.1.6 cannot be found (uncomment if needed). 

!curl -X PURGE https://pypi.org/simple/kaggle-environments



# ConnectX environment was defined in v0.1.6

!pip install 'kaggle-environments>=0.1.6'
from kaggle_environments import evaluate

from kaggle_environments import make

env = make("connectx", debug=True)
def agent_alphabeta(obs, config):



    ################################

    # Imports and helper functions #

    ################################

    

    import numpy as np

    import random



    # Helper function for is_terminal_node: checks if agent or opponent has four in a row in the window

    def is_terminal_window(window, config):

        return window.count(1) == config.inarow or window.count(2) == config.inarow



    # Helper function for minimax_alphabeta: checks if game has ended

    def is_terminal_node(grid, config):

        # Check for draw

        # The list method would be faster.

        #valid_moves = [c for c in range(config.columns) if grid[0][c] == 0]

        #if len(valid_moves) == 0:

        if list(grid[0, :]).count(0) == 0:

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



    # Helper function for count_windows: checks if window satisfies heuristic conditions

    def check_window(window, num_discs, piece, config):

        return (window.count(piece) == num_discs and window.count(0) == config.inarow-num_discs)



    # Helper function for get_heuristic: counts number of windows satisfying specified heuristic conditions

    def count_windows(grid, num_discs, piece, config):

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



    # Helper function for minimax: calculates value of heuristic for grid

    def get_heuristic(grid, mark, config):

        num_threes = count_windows(grid, 3, mark, config)

        num_fours = count_windows(grid, 4, mark, config)

        num_threes_opp = count_windows(grid, 3, mark%2+1, config)

        num_fours_opp = count_windows(grid, 4, mark%2+1, config)

        return num_threes - 1e2*num_threes_opp - 1e4*num_fours_opp + 1e6*num_fours



    # Gets board at next step if agent drops piece in selected column

    def drop_piece(grid, col, mark, config):

        next_grid = grid.copy()

        for row in range(config.rows-1, -1, -1):

            if next_grid[row][col] == 0:

                break

        next_grid[row][col] = mark

        return next_grid



    # Minimax implementation: recursive!!!

    # https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning#Pseudocode

    #def minimax_alphabeta(node, depth, alpha, beta, maximizingPlayer, mark, config):

    def minimax_alphabeta(node, depth, alpha, beta, maximizingPlayer, mark, config, columns_centered): # columns_centered

        if depth == 0 or is_terminal_node(node, config):

            return get_heuristic(node, mark, config)

        #valid_moves = [c for c in range(config.columns) if node[0][c] == 0]

        valid_moves = [c for c in columns_centered if node[0][c] == 0] # columns_centered

        if maximizingPlayer:

            value = -np.Inf

            for col in valid_moves:

                child = drop_piece(node, col, mark, config)

                #value = max(value, minimax_alphabeta(child, depth-1, alpha, beta, False, mark, config))

                value = max(value, minimax_alphabeta(child, depth-1, alpha, beta, False, mark, config, columns_centered)) # columns_centered

                alpha = max(alpha, value) # the minimum score that the maximizing player (i.e., the "alpha" player) is assured of

                if alpha >= beta: #

                    break # beta cutoff

            return value

        else:

            value = np.Inf

            for col in valid_moves:

                child = drop_piece(node, col, mark%2+1, config)

                #value = min(value, minimax_alphabeta(child, depth-1, alpha, beta, True, mark, config))

                value = min(value, minimax_alphabeta(child, depth-1, alpha, beta, True, mark, config, columns_centered)) # columns_centered

                beta = min(beta, value) # the maximum score that the minimizing player (i.e. the "beta" player) is assured of

                if beta <= alpha: #

                    break # alpha cutoff

            return value



    # Uses minimax to calculate value of dropping piece in selected column

    #def score_move(grid, col, mark, config, nsteps):

    def score_move(grid, col, mark, config, nsteps, columns_centered): # columns_centered

        #return minimax_alphabeta(grid, nsteps, -np.Inf, np.Inf, True, mark, config)        # no need to create next_grid here. minimax_alphabeta() will do that.

        # but this method of calculating next_grid is faster!

        next_grid = drop_piece(grid, col, mark, config)

        #return minimax_alphabeta(next_grid, nsteps-1, -np.Inf, np.Inf, False, mark, config)

        return minimax_alphabeta(next_grid, nsteps-1, -np.Inf, np.Inf, False, mark, config, columns_centered) # columns_centered





    #########################

    # Agent makes selection #

    #########################



    # How deep to make the game tree: higher values take longer to run!

    N_STEPS = 2



    # Get list of valid moves

    #valid_moves = [c for c in range(config.columns) if obs.board[c] == 0]

    # columns more center would get higher heuristics. This could be useful in pruning.

    #columns = [c for c in range(config.columns)] # error in v17 why?

    columns = [c for c in range(config.columns) if obs.board[c] == 0]

    dist_from_center = {c: abs(c-(config.columns-1)/2) for c in columns}

    columns_centered = [k for k, v in sorted(dist_from_center.items(), key=lambda item: item[1])]

    # Convert the board to a 2D grid

    grid = np.asarray(obs.board).reshape(config.rows, config.columns)

    # Use the heuristic to assign a score to each possible board in the next step

    #scores = dict(zip(valid_moves, [score_move(grid, col, obs.mark, config, N_STEPS) for col in valid_moves]))

    scores = dict(zip(columns_centered, [score_move(grid, col, obs.mark, config, N_STEPS, columns_centered) for col in columns_centered])) # columns_centered

    # Get a list of columns (moves) that maximize the heuristic

    max_cols = [key for key in scores.keys() if scores[key] == max(scores.values())]

    # Select at random from the maximizing columns

    return random.choice(max_cols)
import inspect

import os



def write_agent_to_file(function, file):

    with open(file, "a" if os.path.exists(file) else "w") as f:

        f.write(inspect.getsource(function))

        print(function, "written to", file)



write_agent_to_file(agent_alphabeta, "submission.py")
import sys

from kaggle_environments import utils



out = sys.stdout

submission = utils.read_file("/kaggle/working/submission.py")

agent = utils.get_last_callable(submission)

sys.stdout = out



env = make("connectx", debug=True)

env.run([agent, agent])

print("Success!" if env.state[0].status == env.state[1].status == "DONE" else "Failed...")