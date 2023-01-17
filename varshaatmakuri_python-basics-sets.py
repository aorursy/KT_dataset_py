a={1,2,3,4};
print(a);
a.add(4);
a.add(5);
print(a);
#Sets do not have duplicates;
#Another way of initializing.
b=set([1,2,1,3,1,3,5,2]);
print(b);
#UNION
a={1,2};
b={3,4};
c=a|b;
print(c);
c1=a.union(b);
print(c1);
#INTERSECTION
a={1,2,3};
b={3,4,5};
c=a&b;
print(c);
c1=a.intersection(b);
print(c1);
#DIFFERENCE
a={1,2,3};
b={3,4,5};
c=a-b;
print(c);
c1=a.difference(b);
print(c1);
#Symmetric difference 
#not common in both
a={1,2,3};
b={3,4,5};
c=a^b;
print(c);
c1=a.symmetric_difference(b);
print(c1);
#SUBSET, SUPERSET
a={1,2,3,4,5};
b={2,3};

print(a.issubset(b));
print(b.issubset(a));
print(b<=a);

print(a.issuperset(b));
print(b.issuperset(a));
print(a<=b);
#Frozenset
a=frozenset([1,2]);
print(a);

a.add(99);
#Not supported. Frozen set can be modified.

