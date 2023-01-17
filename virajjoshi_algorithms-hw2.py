def fib(n):

    """Slow, recursive version"""

    if n < 2: return n

    return fib(n-1) + fib(n-2)
for i in range(10):

    print(i, fib(i))
def fib2(n):

    cache = {}



    def fib2_inner(n):

        """Memoize the recursion"""

        if n < 2: return n

        if n in cache:

            return cache[n]

        ans = fib2_inner(n-1) + fib2_inner(n-2)

        cache[n] = ans

        return ans



    return fib2_inner(n)
for i in range(10):

    print(i, fib2(i))
# of any function.

import functools



def memoize(f):

    """This can turn any recursive function into a memoized version."""

    cache = {}

    @functools.wraps(f)

    def wrap(*args):

        if args not in cache:

            cache[args] = f(*args)  

        return cache[args]

    return wrap



@memoize

def fib2a(n):

    """Same code as the recursive version -- but faster!"""

    if n < 2: return n

    return fib2a(n-1) + fib2a(n-2)
for i in range(10):

    print(i, fib2a(i))
# Let's see how long these take

    

print("Without memoization:")

%timeit fib(30)



print("With memoization:")

%timeit fib2(30)



print("With memoization as a decorator:")

%timeit fib2a(30)
# # Let's try a big number!

# fib2(3000)

# # (this should give a RecursionError)
import sys

import time as time

sys.setrecursionlimit(100000)

t=time.time()

fib2(3000)

print("Memoization, n=3,000:")

%timeit fib2(3000)

# This should work!  If not, lower it to 1000.
# # This probably doesn't, though:

# fib2(50000)

# # If you run this, you probably crash the kernel and need to

# # rerun everything above this.  Then skip this next time!
# An alternative solution is iterative

# Better because python doesn't have much space on the stack

def fib3(n):

    """Bottom-up iterative solution"""

    if n == 0 or n==1:

        return n

    

    fibs = [1, 1] + [0]*n

    for i in range(2, n+1):

            fibs[i] = fibs[i-1] + fibs[i-2]

    return fibs[n]
for i in range(10):

    print(i, fib3(i))
print("n = 30:")

%timeit fib3(30)

print("n = 1000:")

%timeit fib3(1000)

print("n = 10000:")

%timeit fib3(10000)

print("n = 100000:")

%timeit fib3(100000)
def fib4(n):

    """Sliding window version, improves space complexity"""

    if n < 2:

        return 1

    

    a1 = 1

    a2 = 1



    for i in range(2, n+1): 

        a1, a2 = a2, a1+a2

    return a2
for i in range(10):

    print(i, fib2(i))
print("n = 30:")

%timeit fib4(30)

print("n = 10000:")

%timeit fib4(10000)

print("n = 100000:")

%timeit fib4(100000)
import numpy as np



def fib5(n):

    """Use matrix exponentiation to do it faster"""

    # Set dtype=object to use Python longs, so it doesn't overflow

    A = np.matrix([[1,1],

                   [1,0]], dtype='object')

    return (A**n)[0,1]
for i in range(10):

    print(i, fib5(i))
print("n = 30:")

%timeit fib5(30)

print("n = 10000:")

%timeit fib5(10000)

print("n = 100000:")

%timeit fib5(100000)
# Let's get a timing function that's easier to work with.



# The tricky bit is that the clock is only reliable at large scales (> .1 seconds, say)

# so we need to run the function many times; but the number of times depends on the input.





# The %timeit magic in jupyter does this, but it doesn't have a nice interface.

# Instead we use exponential search on the basic timeit module.



import timeit



def smarter_timeit(f, min_total_time=0.1):

    """Figure out how long f() takes to run.

    

    To make the timing reliable, run the function enough times to take min_time seconds overall.

    """

    number = 1

    while True:

        total_time = timeit.timeit(f, number=number)

        if total_time > min_total_time:

            return total_time / number

        number *= 2
smarter_timeit(lambda: fib5(200000))
# Get all the data

def get_times(func, base=1.01, max_val=None, max_time=0.2, min_time=0.1):

    """Get the running time for a given function at many different n.

    

    Gets the time to compute func(n) for n growing up until it takes

    max_time seconds to compute func(n), and return a pair of lists:

    the times and corresponding ns. 

    """

    nums = []

    vals = []

    for i in range(1000):

        n = int(base**i)

        if nums and n == nums[-1]:

            continue

        if max_val is not None and n > max_val:

            break

        t = smarter_timeit(lambda: func(n), min_time)

        nums.append(n)

        vals.append(t)

        if t > max_time:

            break

    return nums, vals
import matplotlib.pyplot as plt

%matplotlib inline
recursive = get_times(fib, 1.01)

plt.plot(*recursive, marker='o')

plt.ylabel("Time (seconds)")

plt.xlabel("n")
# Because fib2 hits the recursion limit, we stop at n=1000.

memoized = get_times(fib2, 1.2, max_val=1000)

plt.plot(*memoized, marker='o')
# All the following plots should give smooth curves;

# if they don't, try running a second time.  Random

# hiccups can happen, since we don't run the large

# values many times.  They also take a few seconds to

# run.



iterative = get_times(fib3, 1.2)

plt.plot(*iterative, marker='o')
# Let's try overlaying a quadratic that matches at the largest n

def fit_curve(p, nums, times):

    xx = np.linspace(nums[0], nums[-1], 1000)

    yy = times[-1] / xx[-1]**p * xx**p

    return xx, yy



plt.plot(*iterative, marker='o')

plt.plot(*fit_curve(2, *iterative))
sliding = get_times(fib4, 1.2)

plt.plot(*sliding, marker='o')

plt.plot(*fit_curve(2, *sliding))
import math as math

matrix = get_times(fib5, 1.2)

plt.plot(*matrix, marker='o')

# The quadratic doesn't fit well

plt.plot(*fit_curve(2, *matrix), label='p=2')

# The power is log_2 3 instead

plt.plot(*fit_curve(math.log(3)/math.log(2), *matrix), label='p=1.585')

plt.legend()
# Let's put all the plots together

plt.plot(*iterative, marker='o', label='Iterative')

plt.plot(*fit_curve(2, *iterative))

plt.plot(*sliding, marker='o', label='Sliding')

plt.plot(*fit_curve(2, *sliding))

plt.plot(*matrix, marker='o', label='Matrix')

plt.plot(*fit_curve(math.log(3)/math.log(2), *matrix))

plt.legend()
# On a log-log plot, n^p becomes a straight lines, and the slope gives the exponent.



plt.plot(*iterative, label='Iterative')

plt.plot(*fit_curve(2, *iterative), ls='--')

plt.plot(*sliding, label='Sliding')

plt.plot(*fit_curve(2, *sliding), ls='--')

plt.plot(*matrix, label='Matrix',color="orange")

plt.loglog(*fit_curve(math.log(3)/math.log(2), *matrix), ls='--')

plt.axis('scaled')

plt.ylim((10**(-7), None))

plt.legend()
# Let's use GMP rather than python long ints

# This requires the gmpy library,

# which is installable with `pip3 install gmpy` (or perhaps, `pip install gmpy`)



import gmpy



def fib6(n):

    """Same code as fib5, but with gmpy objects instead of Python longs."""

    one = gmpy.mpz(1)

    A = np.matrix([[one,one],[one,one*0]], dtype='object')

    v = (A**n).dot([[1],[0]])

    return v[0,0]
gmp_matrix = get_times(fib6, 1.4, max_time=0.2)

plt.plot(*gmp_matrix, marker='o')
# How does it seem on a log log plot?

plt.plot(*iterative, label='Iterative')

#plt.plot(*fit_curve(2, *iterative), ls='--')

plt.plot(*sliding, label='Sliding')

#plt.plot(*fit_curve(2, *sliding), ls='--')

plt.plot(*matrix, label='Matrix')

#plt.plot(*fit_curve(math.log(3)/math.log(2), *matrix), ls='--')

plt.loglog(*gmp_matrix, label='GMP matrix')

plt.ylabel("Time (seconds)")

plt.xlabel("n")

plt.legend()