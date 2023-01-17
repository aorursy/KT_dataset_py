# Create a 3x3 tic tac toe board of "" strings for each value
#board = [' ' for i in range(9)]
# Create a function to display your board
    
def display_board(board):
    print()
    print('   |   |')
    print(' ' + board[0] + ' | ' + board[1] + ' | ' + board[2])
    print('   |   |')
    print('-----------')
    print('   |   |')
    print(' ' + board[3] + ' | ' + board[4] + ' | ' + board[5])
    print('   |   |')
    print('-----------')
    print('   |   |')
    print(' ' + board[6] + ' | ' + board[7] + ' | ' + board[8])
    print('   |   |') 
    print()
    print()
#Create a function to check if anyone won, Use marks "X" or "O"
def check_win(mark, board):
#     print(board)
    
    return ((board[6] == mark and board[7] == mark and board[8] == mark) or # across the top
    (board[3] == mark and board[4] == mark and board[5] == mark) or # across the middle
    (board[0] == mark and board[1] == mark and board[2] == mark) or # across the bottom 
    (board[7] == mark and board[4] == mark and board[1] == mark) or # down the middle
    (board[8] == mark and board[5] == mark and board[2] == mark) or
    (board[6] == mark and board[3] == mark and board[0] == mark) or# down the right side
    (board[6] == mark and board[4] == mark and board[2] == mark) or # diagonal
    (board[8] == mark and board[4] == mark and board[0] == mark)) # diagonal

# Create a function to check its a Draw
def check_draw(board):
        return ' ' not in board
# Create a Function that makes a copy of the board
def board_copy(board):
    newBoard = [item for item in board]
    return newBoard
#Immediate move checker
def test_win_move(move, player_mark, board):
    if(board[move] == ' '):
        board[move] = player_mark
        return check_win(player_mark, board)
    else:
        print('Invalid possition')
        return False
    
#Strategy if others fail
def win_strategy(board):
    print(board)
    display_board(board)
    if(board[4]) == ' ':
        return 4
    for i in [0, 2, 6, 8]:
        if(board[i] == ' '):
            return i
    for i in [1, 3, 5, 7]:
        if(board[i] == ' '):
            return i
# Agents move
def get_agent_move(board):
    # Return agent move with your strategy
    bCopy = board_copy(board)
    for i in range(0, 9):
        if(board[i] == ' ' and test_win_move(i, 'O', bCopy)):
            return i
    for i in range(0, 9):
        if(board[i] == ' ' and test_win_move(i, 'X', bCopy)):
            return i



        
    return win_strategy(board)
# Assemble the game
def tictactoe():
    ### Note you need to recreate your board again here if you wish to play the game more than once
    board = [' ' for i in range(9)]
    print('Welcome')
    while(not check_draw(board)):
        display_board(board)
        move = int(input('Your turn! Enter position : '))
        checkWin = test_win_move(move, 'X', board)
        print(checkWin)
        checkDraw = check_draw(board)
        display_board(board)
        if(checkWin):
            print('You won  !')
            display_board(board)
            break
        if(checkDraw):
            print('Draw...')
            display_board(board)
            break
        print('Agent thinking....')
        agentMove = get_agent_move(board)
        checkWin = test_win_move(agentMove, 'O', board)
        checkDraw = check_draw(board)
        if(checkWin):
            print('You lost  !')
            display_board(board)
            break
        if(checkDraw):
            print('Draw...')
            display_board(board)
            break
    
# Play!!!
tictactoe()
class Tic_Tac_Toe:
    def __init__(self):
        self.board = [' ' for i in range(9)]
        
    def display_board(self, board):
        print()
        print('   |   |')
        print(' ' + board[0] + ' | ' + board[1] + ' | ' + board[2])
        print('   |   |')
        print('-----------')
        print('   |   |')
        print(' ' + board[3] + ' | ' + board[4] + ' | ' + board[5])
        print('   |   |')
        print('-----------')
        print('   |   |')
        print(' ' + board[6] + ' | ' + board[7] + ' | ' + board[8])
        print('   |   |') 
        print()
        print()
        
    #Create a function to check if anyone won, Use marks "X" or "O"
    def check_win(self, mark, board):
        return ((board[6] == mark and board[7] == mark and board[8] == mark) or # across the top
        (board[3] == mark and board[4] == mark and board[5] == mark) or # across the middle
        (board[0] == mark and board[1] == mark and board[2] == mark) or # across the bottom
        (board[7] == mark and board[4] == mark and board[1] == mark) or # down the middle
        (board[8] == mark and board[5] == mark and board[2] == mark) or
        (board[6] == mark and board[3] == mark and board[0] == mark) or# down the right side
        (board[6] == mark and board[4] == mark and board[2] == mark) or # diagonal
        (board[8] == mark and board[4] == mark and board[0] == mark)) # diagonal
    
    # Create a function to check its a Draw
    def check_draw(self, board):
        return ' ' not in board
    

    # Create a Function that makes a copy of the board
    def board_copy(self, board):
        newBoard = [item for item in board]
        return newBoard
    
    def test_win_move(self, move, player_mark, board):
        if(board[move] == ' '):
            board[move] = player_mark
            return self.check_win(player_mark, board)
        else:
            print('Invalid possition')
            return False
    
    #Strategy if others fail
    def win_strategy(self, board):
        if(board[4]) == ' ':
            return 4
        for i in [0, 2, 6, 8]:
            if(board[i] == ' '):
                return i
        for i in [1, 3, 5, 7]:
            if(board[i] == ' '):
                return i
            
    # Agents move
    def get_agent_move(self, board):
        # Return agent move with your strategy
        bCopy = self.board_copy(board)
        for i in range(0, 9):
            if(board[i] == ' ' and self.test_win_move(i, 'O', bCopy)):
                return i
        for i in range(0, 9):
            if(board[i] == ' ' and self.test_win_move(i, 'X', bCopy)):
                return i
            
        return win_strategy(board)
    
    # Assemble the game
    def tictactoe(self):
        ### Note you need to recreate your board again here if you wish to play the game more than once
        
        print('Welcome')
        while(not self.check_draw(self.board)):
            self.display_board(self.board)
            move = int(input('Your turn! Enter position : '))
            checkWin = self.test_win_move(move, 'X', self.board)
            checkDraw = self.check_draw(self.board)
            self.display_board(self.board)
            if(checkWin):
                print('You won  !')
                self.display_board(self.board)
                break
            if(checkDraw):
                print('Draw...')
                self.display_board(self.board)
                break
            print('Agent thinking....')
            agentMove = self.get_agent_move(self.board)
            checkWin = self.test_win_move(agentMove, 'O', self.board)
            checkDraw = self.check_draw(self.board)
            if(checkWin):
                print('You lost  !')
                self.display_board(self.board)
                break
            if(checkDraw):
                print('Draw...')
                self.display_board(self.board)
                break
    
    
newGame = Tic_Tac_Toe()
newGame.tictactoe()
