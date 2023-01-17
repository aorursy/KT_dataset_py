
# Create a function to display your board
def display_board(board):
    print('   |   |  ')
    print(' ' + board[0]+' | '+board[1]+' | '+board[2])
    print('___|___|___')
    print('   |   |  ')
    print(' ' + board[3]+' | '+board[4]+' | '+board[5])
    print('___|___|___')
    print('   |   |  ')
    print(' ' + board[6]+' | '+board[7]+' | '+board[8])
    print('   |   |   ')
display_board(board)
#Create a function to check if anyone won, Use marks "X" or "O"
def check_win(player_mark, board):
    ## If the player has won then there must be 3 consecutive Player values
    return ((board[0]==player_mark and board[1]==player_mark and board[2]==player_mark) or
      (board[3]==player_mark and board[4]==player_mark and board[5]==player_mark) or
      (board[6]==player_mark and board[7]==player_mark and board[8]==player_mark) or
      (board[0]==player_mark and board[3]==player_mark and board[6]==player_mark) or
      (board[1]==player_mark and board[4]==player_mark and board[7]==player_mark) or
      (board[2]==player_mark and board[5]==player_mark and board[8]==player_mark) or
      (board[0]==player_mark and board[4]==player_mark and board[8]==player_mark) or
      (board[2]==player_mark and board[4]==player_mark and board[6]==player_mark))

# check_win('X', board)
# Create a function to check its a Draw
def check_draw(board):
    return ' ' not in board
# Create a Function that makes a copy of the board
def board_copy(board):
    new_board = []
    for i in board: new_board.append(i)
    return new_board
#Immediate move checker
def test_win_move(move, player_mark, board):
    bCopy = board_copy(board)
    bCopy[move] = player_mark
    check_win(player_mark, board)
#Strategy if others fail
def win_strategy(board):
    if board[4] == ' ': return 4
    for i in [0, 2, 6, 8]: 
        if board[i] == ' ': return i
    for i in [1, 3, 5, 7]: 
        if board[i] == ' ': return i
    
def get_player_move(board):
    run = True
    while run:
        move = int(input('Please select a position'))
        if board[move] == ' ':
            run = False
            board[move] == 'X'
        else: print('Sorry the place is occupied')
    
# Agents move
def get_agent_move(board):
    # Return agent move with your strategy
    for i in range(0,9):
        if board[i] == ' ' and test_win_move(board, 'X', i): return i
    for i in range(0,9):
        if board[i] == ' ' and test_win_move(board, 'O', i): return i
    
    return win_strategy(board)
# Assemble the game
def tictactoe(board):
    ### Note you need to recreate your board again here if you wish to play the game more than once
    print('Welcome to Tic Tac Toe')
    displayBoard(board)
    
    while(' ' in board):
        if not(check_win('O', board)):
            get_player_move(board)
            display_board(board)
        else:
            print('Computer has won! Better luck next time.')
            break
        if not(check_win('X', board)):
            get_agent_move(board)
            display_board(board)
        else:
            print('You have won! Good job.')
            break
    if ' ' not in board: print('Tie')
# Play!!!
board = ' '*9
tictactoe(board)
class Tic_Tac_Toe:
    def __init__(self):
        pass