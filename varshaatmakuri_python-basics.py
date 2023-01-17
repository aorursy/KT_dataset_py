#Defining function
def add5(no) :
    return no + 5;
print(add5(3));

#Function with default aruguments.
def printme(a='Hello'):
    print(a);
printme('Hi');

printme();
# Varying length of aruments
def sum(*a):
    b=0;
    for i in a:
        b=b+i;
        
    return b;

print(sum(1,2,3,4));
#LAMBDA
add2 = lambda x : x+2;
print(add2(1));

b=map( add2, (1,2,3));
print(list(b));
import functools;
print(functools.reduce(lambda x,y : x+y, [1,2,3]));
print(functools.reduce(lambda x,y : x if x > y else y, (4,6,1,2,3)));
