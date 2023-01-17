import random
def change_size_recursive(denominations, amount):

    """Find the smallest number of bills required to make change

    for amount, if each bill is one of the given denominations.

    

    Input:

        denominations: a list of integer bill sizes (e.g., [2,3,13])

        amount: the amount you want to make change for

    

    Output:

        A single integer, the fewest bills possible to make exact change of size amount.

        If no solution is possible, return None.

    """

    def f(amount):

        if amount == 0:

            return 0

        if amount < 0:

            return None

        best_option = None

        for bill in denominations:

            option = f(amount-bill)

            if option is not None:

                if best_option is None or option < best_option:

                    best_option = option + 1

        return best_option

    return f(amount)

denominations1 = [1, 2, 5, 10, 20, 50, 100]

print(change_size_recursive(denominations1, 16)) #3



denominations2 = [5, 9, 13, 17]

print(change_size_recursive(denominations2, 14)) #2

print(change_size_recursive(denominations2, 16)) #None
# Let's try the same `memoize` as pset 1

# (slightly modified to work with mutable inputs/outputs)

import functools

import pickle



def memoize(f):

    """This can turn any recursive function into a memoized version.

    

    We use pickle.dumps/pickle.loads to deal with mutable inputs/outputs.

    If the function returns a list, we don't want a subsequent modification

    (e.g., result.append(1)) to corrupt the cached value.

    """

    cache = {}

    @functools.wraps(f)

    def wrap(*args):

        args_hash = pickle.dumps(args)

        if args_hash not in cache:

            cache[args_hash] = pickle.dumps(f(*args))  

        return pickle.loads(cache[args_hash])

    wrap.cache = cache                #so we can clear it as needed

    return wrap
#The code is exactly the same as change_size_recursive, but with @memoize.

def change_size_memoized(denominations, amount):

    """Find the smallest number of bills required to make change

    for amount, if each bill is one of the given denominations.

    

    Input:

        denominations: a list of integer bill sizes (e.g., [2,3,13])

        amount: the amount you want to make change for

    

    Output:

        A single integer, the fewest bills possible to make exact change of size amount.

        If no solution is possible, return None.

    """

    @memoize

    def f(amount):

        if amount == 0:

            return 0

        if amount < 0:

            return None

        best_option = None

        for bill in denominations:

            option = f(amount-bill)

            if option is not None:

                if best_option is None or option < best_option:

                    best_option = option + 1

        return best_option

    return f(amount)
denominations1 = [1, 2, 5, 10, 20, 50, 100]

print(change_size_memoized(denominations1, 16)) #3

print(change_size_memoized(denominations1, 406)) #6



denominations2 = [5, 9, 13, 17]

print(change_size_memoized(denominations2, 14)) #2

print(change_size_memoized(denominations2, 16)) #None

print(change_size_memoized(denominations2, 1006)) #62



denominations3 = random.sample(range(100, 200), 10)



%timeit change_size_memoized(denominations1, 406)

%timeit change_size_memoized(denominations3, 4006)

%timeit change_size_memoized(denominations3, 40006)
# Here's one way: we modify change_set_memoized a bit.



def change_set_memoized(denominations, amount):

    """Find the smallest number of bills required to make change

    for amount, if each bill is one of the given denominations.

    

    Input:

        denominations: a list of integer bill sizes (e.g., [2,3,13])

        amount: the amount you want to make change for

    

    Output:

        A list of bill values, the shortest possible such list with sum `amount`.

        If no solution is possible, return None.

    """

    @memoize

    def f(amount):

        if amount == 0:

            return []

        if amount < 0:

            return None

        best_option = None

        for bill in denominations:

            option = f(amount-bill)

            if option is not None:

                if best_option is None or sum(option) < sum(best_option):

                    best_option = option+[bill]

        return best_option

    return f(amount)
print(change_set_memoized(denominations1, 16))   # [1, 5, 10]

print(change_set_memoized(denominations1, 406))  # [1, 5, 100, 100, 100, 100]



print(change_set_memoized(denominations2, 14))   # [5, 9]

print(change_set_memoized(denominations2, 16))   # None

print(change_set_memoized(denominations2, 1006)) # [5, 5, 5, 5, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17]



%timeit change_set_memoized(denominations1, 406)

%timeit change_set_memoized(denominations3, 4006)

%timeit change_set_memoized(denominations3, 40006)
# As you can see, this is essentially the same

# code as in `change_size_recursive` and all the other

# versions, just run in a different order.

def change_bottomup_size(denominations, amount):

    best = [None]*(amount+1)

    best[0] = 0

    for i in range(1, amount+1):

        best_option = None

        for bill in denominations:

            if bill <= i and best[i-bill] is not None:

                option = best[i-bill] + 1

                if best_option is None or option < best_option:

                    best_option = option

        best[i] = best_option

    return best[amount]

denominations1 = [1, 2, 5, 10, 20, 50, 100]

print(change_bottomup_size(denominations1, 16)) #3

print(change_bottomup_size(denominations1, 406)) #6



denominations2 = [5, 9, 13, 17]

print(change_bottomup_size(denominations2, 14)) #2

print(change_bottomup_size(denominations2, 16)) #None

print(change_bottomup_size(denominations2, 1006)) #62



%timeit change_bottomup_size(denominations1, 406)

%timeit change_bottomup_size(denominations3, 4006)

%timeit change_bottomup_size(denominations3, 40006)
def change_bottomup_set(denominations, amount):

    best = [None]*(amount+1)

    best[0] = 0

    back = [None]*(amount+1)                   # new line

    for i in range(1, amount+1):

        best_option = None

        for bill in denominations:

            if bill <= i and best[i-bill] is not None:

                option = best[i-bill] + 1

                if best_option is None or option < best_option:

                    best_option = option

                    back[i] = bill             # new line

        best[i] = best_option

        

    # Now that we have the back[] array, produce the answer.

    if best[amount] is None:

        return None

    answer = []

    remaining = amount

    while remaining:

        answer.append(back[remaining])

        remaining -= back[remaining]

    return answer
denominations1 = [1, 2, 5, 10, 20, 50, 100]

print(change_bottomup_set(denominations1, 16)) #[1, 5, 10]

print(change_bottomup_set(denominations1, 406)) # [1, 5, 100, 100, 100, 100]



denominations2 = [5, 9, 13, 17]

print(change_bottomup_set(denominations2, 14)) # [5, 9]

print(change_bottomup_set(denominations2, 16)) # None

print(change_bottomup_set(denominations2, 1006)) # [5, 5, 5, 5, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17]



%timeit change_bottomup_set(denominations1, 406)

%timeit change_bottomup_set(denominations3, 4006)

%timeit change_bottomup_set(denominations3, 40006)