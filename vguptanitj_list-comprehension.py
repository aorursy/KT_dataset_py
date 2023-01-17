nums_plus_one = [ num**2 for num in range(11) ]

print(nums_plus_one)
multiple_of_5 = [num for num in range (5,100,5)]

print(multiple_of_5)
product = ["Toy","Dress","Book","Copy","Bag"]

price = [500,1000,2000,300,500]



product_with_price = [ (product[i],price[i]) for i in range(len(product)) ]



print(product_with_price)
matrix = [[col for col in range (3)] for row in range (4)]

print(matrix)
odd = [num for num in range (10) if num % 2 == 1]

print(odd)
products = [ "Toy","Dress","Book","Copy","Bag","Pen" ,"Pencil","Bottle","Shoes" ]

wishlist = ["Eraser","Ruler","Pen","Book","Copy"]



avalaible_list = ["Yes" if item in products else "No" for item in wishlist]



print(avalaible_list)