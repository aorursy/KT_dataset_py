#Q1.

def is_multiple(n, m):

    return n % m == 0
#Q2.

[2**i for i in range(9)]
#Q3.

def is_list_unique(list):

    return len(set(list)) == len(list)
#Q4.

def harmonic_gen(n):

    h = 0 

    for i in range(1, n + 1): 

        h += 1 / i 

    yield h 