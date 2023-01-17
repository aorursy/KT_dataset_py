def G_n(state):

    # Note: You can skip this step if you feel you have a better way of getting G(n)

    pass
def H_n(state, target, g):

    #Manhattan distance of each move

    cost = 0

    for i in state:

        d1, d2 = state.index(i), target.index(i)

        p, q = d1 % 3, d1 // 3

        x, y = d2 % 3, d2 //3

        cost+= abs(p-x) + abs(q-y)

    return cost+g
def F_n(poss_moves,target,g):# Fill inputs as necessary

    next_move = [(i,H_n(i,target,g)) for i in poss_moves] #Call H_n to get distance for each move i in poss_moves

    next_move = sorted(next_move, key = lambda x : x[1]) #Sort tuple according to distance

    return [i for i,j in next_move if j == next_move[0][1]] #return all moves with min distance

    
def astar(state, target):# Add inputs if more are required

    visited_states=[]

    visited_states.append(src)

    g = 0

    arr = [src]

    c = 0

    while arr:

        c += 1                                           # Calculate Number of Iterations                                   # Print current state to check

        if arr[0] == target:                             # break if target found

            return True

        arr += F_n(possible_moves(arr[0],visited_states),target,g)  #Call F_n to get next_move

        visited_states.append(arr[0]) 

        print("Level {} : {}".format(g,arr[0]))

        arr.pop(0)                                       # remove checked move from arr 

        g+=1

    return False
def possible_moves(state,visited_states):# Add inputs if more are required

    ind = state.index(-1)

    d=[]

    if ind+3 in range(9): d.append('d')

    if ind-3 in range(9): d.append('u')

    if ind not in [0,3,6]: d.append('l')

    if ind not in [2,5,8]: d.append('r')

    pos_moves=[]

    for move in d:

        pos_moves.append(gen(state,move,ind))

    return [move for move in pos_moves if move not in visited_states]

def gen(state, direction, b):

    temp=state.copy()

    if direction == 'd': temp[b+3] , temp[b] = temp[b], temp[b+3]

    if direction == 'u': temp[b-3] , temp[b] = temp[b], temp[b-3]

    if direction == 'l': temp[b-1] , temp[b] = temp[b], temp[b-1]

    if direction == 'r': temp[b+1] , temp[b] = temp[b], temp[b+1]

    return temp
#Test 1

src = [1,2,3,-1,4,5,6,7,8]

target = [1,2,3,4,5,-1,6,7,8]         





astar(src, target) 
# Test 2

src = [1,2,3,-1,4,5,6,7,8] 

target=[1,2,3,6,4,5,-1,7,8]







astar(src, target)
# # Test 3

# src = [1,2,3,7,4,5,6,-1,8] 

# target=[1,2,3,6,4,5,-1,7,8]







# astar(src, target)