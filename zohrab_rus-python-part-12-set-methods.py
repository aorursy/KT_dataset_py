st_1 = {1, 10, 30, 40}
st_2 = {7, 10, 12}
st_1.intersection(st_2)
st_1 & st_2
st_1 and st_2
st_1.intersection_update({7, 10})
st_1
st_1 = {1, 7, 10}
st_2 = {7, 10, 11, 30, 31, 40, 41}
st_1.union(st_2)
st_1 | st_2
st_1 or st_2
st_1.difference(st_2)
st_1 - st_2
st_1.difference_update({(6,6,7), 1, 40})
st_1
st_2 = {7, 15, 20}
st_1.symmetric_difference(st_2)
st_1 ^ st_2
st_1.symmetric_difference_update({7, 15, 20})
st_1
st_2.issubset(st_1)
st_2 = {10, 15}
st_2.issubset(st_1)
st_1.issuperset(st_2)
st_2 = {7, 10, 12}
st_1.issuperset(st_2)
st_1.isdisjoint(st_2)
st_2 = {12}
st_1.isdisjoint(st_2)
fset = frozenset([0, 2, 2, 1, 1, 0])
fset
fset.add(3)
'copy',

'difference',

'intersection',

'isdisjoint',

'issubset',

'issuperset',

'symmetric_difference',

'union'