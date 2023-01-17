def gib_folgezustaende(zustand, max_people, boat_size):
    m,k,b = zustand
    incr = -1 if b else +1
    
    # Just go through all possibilities and filter the ones that wouldn't fit into the boat.
    fzustaende = []
    for i in range(0, max_people + 1):
        for j in range(0, max_people + 1):
            if i + j <= boat_size and i + j > 0: fzustaende.append([m + i * incr, k + j * incr, not b])
            
    # Remove all that are negative or greater than the maximum.
    return [[k,m,b] for k,m,b in fzustaende 
                if k >= 0 and k <= max_people and m >= 0 and m <= max_people]
def is_valid(zustand, max_value):
    m,k,b = zustand
    # es gibt im Westen mehr Kannibalen, als Missionare
    if m < k and m > 0: return False 
    # es gibt im Osten mehr Kannibalen, als Missionare
    if m > k and m < max_value: return False
    return True

def gib_valide_folgezustaende(zustand, max_people, boat_size):
    return [z for z in gib_folgezustaende(zustand, max_people, boat_size) if is_valid(z, max_people)]
# Rekursive Suche in die Tiefe (depth-first search with chronolocigal backtracking)
def suche(zustand, max_people, boat_size, history=[], all_solutions=False, level=0, debug=0):
    if debug: print(level*' ',zustand," ->",end="")
        
    if zustand == [0,0,False]: return (True,history+[zustand])
    fzustaende = gib_valide_folgezustaende(zustand, max_people, boat_size)
    
    if debug: print("  ",fzustaende)
        
    if not len(fzustaende): return (False,[])
    for z in fzustaende:
        if z not in history+zustand:
            res1,res2 = suche(z,max_people,boat_size,history+[zustand],all_solutions,level+1,debug)
            if res1: 
                if all_solutions:
                    print("Solution found: ",res1,res2)
                else:
                    return (res1,res2) # Just stop
        else:
            if debug == 2: print((level+1)*' '+"repeated",z)
    return (False,[])
suche([3,3,True], 3, 2)
suche([4,4,1], 4, 3)
for n in range(5, 21): print("Found", [n,n,True], ":", suche([n,n,True], n, n - 1)[0])