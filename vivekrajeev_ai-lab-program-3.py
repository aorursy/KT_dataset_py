# Bredth first search in search of target - Using Brute Force Algorithms



def bfs(src,target):

    que = [src]

    visited_states = []

    while len(que):

        state = que.pop(0)

        print(state)

        visited_states.append(tuple(state))

        if state == target:

            return

        for move in possible_moves(state, visited_states):

            if move not in que and tuple(move) not in visited_states:

                que.append(move)
# Find Possible Moves

def possible_moves(state, visited_states): 

    # Find index of empty spot and assign it to b

    b = state.index(-1);

    

    #'d' for down, 'u' for up, 'r' for right, 'l' for left - directions array

    d = []

                                    

    #Add all possible direction into directions array - Hint using if statements

    if b - 3 in range(9): 

        d.append('u')

    if b not in [0,3,6]: 

        d.append('l')

    if b not in [2,5,8]: 

        d.append('r')

    if b + 3 in range(9): 

        d.append('d')

    

    # If direction is possible then add state to move

    pos_moves = []

    

    # for all possible directions find the state if that move is played

    ### Jump to gen function to generate all possible moves in the given directions

    for m in d:

        pos_moves.append(gen(state, m, b))

    

    # return all possible moves only if the move not in visited_states

    return [move for move in pos_moves if move not in visited_states]



# Generate move for given direction

def gen(state, m, b):

    # m(move) is direction to slide, b(blank) is index of empty spot

    # create a copy of current state to test the move

    temp = state.copy()                              

    

    # if move is to slide empty spot to the left and so on

    if m == 'u': temp[b-3] , temp[b] = temp[b], temp[b-3]

    if m == 'l': temp[b-1] , temp[b] = temp[b], temp[b-1]

    if m == 'r': temp[b+1] , temp[b] = temp[b], temp[b+1]

    if m == 'd': temp[b+3] , temp[b] = temp[b], temp[b+3]   

    

    # return new state with tested move to later check if "src == target"

    return temp
#Test 1

src = [1,2,3,-1,4,5,6,7,8]

target = [1,2,3,4,5,-1,6,7,8]         

       





bfs(src, target) 
# Test 2

src = [1,2,3,-1,4,5,6,7,8] 

target=[1,2,3,6,4,5,-1,7,8]







bfs(src, target)