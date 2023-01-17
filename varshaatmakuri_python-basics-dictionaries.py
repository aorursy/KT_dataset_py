a={'a':1,'b':2};
print('Printing a');
print(a);
print(a.values());
print(a.keys());
#Another way of initializing dictionary elements.
a['c']=3;
a['d']=4;

print('Printing a');
print(a);
#Accessing Elements
print(a['c']);
#Updating and deleting.
print('Printing a');
print(a);

a['c']=33;
del(a['d']);

print('After Updating and deleting');
print('Printing a');
print(a);
for i in a:
    print(i,a[i]);
#Another way of initializing dictionary.
d=dict(a=1,b=2);
print(d);

