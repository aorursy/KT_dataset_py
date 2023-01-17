2 + 3 + 9
99 - 73
23.54 * -1432
100 / 7
100 // 7
100 % 7
5 ** 3
((2 + 5) * (17 - 3)) / (4 ** 3)
500 * (.2 * 1.25)
cost_of_ice_bag = 1.25
profit_margin = .2
number_of_bags = 500
profit_margin
profit_per_bag = cost_of_ice_bag * profit_margin
profit_per_bag
total_profit = number_of_bags * profit_per_bag
total_profit
# net_profit # Shows error here
print("The grocery store makes a total profit of $", total_profit)
# Store input data in variables

cost_of_ice_bag = 1.25

profit_margin = .2

number_of_bags = 500



# Perform the required calculations

profit_per_bag = cost_of_ice_bag * profit_margin

total_profit = number_of_bags * profit_per_bag



# Display the result

print("The grocery store makes a total profit of $", total_profit)
my_favorite_number = 1 # an inline comment
# This comment gets its own line

my_least_favorite_number = 3
"""This is a multi-line comment.

Write as little or as much as you'd like.



Comments are really helpful for people reading

your code, but try to keep them short & to-the-point.



Also, if you use good variable names, then your code is

often self explanatory, and you may not even need comments!

"""

a_neutral_number = 5
my_favorite_number = 1

my_least_favorite_number = 5

a_neutral_number = 3
# Equality check - True

my_favorite_number == 1
# Equality check - False

my_favorite_number == my_least_favorite_number
# Not equal check - True

my_favorite_number != a_neutral_number
# Not equal check - False

a_neutral_number != 3
# Greater than check - True

my_least_favorite_number > a_neutral_number
# Greater than check - False

my_favorite_number > my_least_favorite_number
# Less than check - True

my_favorite_number < 10
# Less than check - False

my_least_favorite_number < my_favorite_number
# Greater than or equal check - True

my_favorite_number >= 1
# Greater than or equal check - False

my_favorite_number >= 3
# Less than or equal check - True

3 + 6 <= 9
# Less than or equal check - False

my_favorite_number + a_neutral_number <= 3
cost_of_ice_bag = 1.25

is_ice_bag_expensive = cost_of_ice_bag >= 10

print("Is the ice bag expensive?", is_ice_bag_expensive)
my_favorite_number
my_favorite_number > 0 and my_favorite_number <= 3
my_favorite_number < 0 and my_favorite_number <= 3
my_favorite_number > 0 and my_favorite_number >= 3
True and False
True and True
a_neutral_number = 3
a_neutral_number == 3 or my_favorite_number < 0
a_neutral_number != 3 or my_favorite_number < 0
my_favorite_number < 0 or True
False or False
not a_neutral_number == 3
not my_favorite_number < 0
not False
not True
(2 > 3 and 4 <= 5) or not (my_favorite_number < 0 and True)
not (True and 0 < 1) or (False and True)
not True and 0 < 1 or False and True
!pip install jovian --upgrade --quiet
import jovian
jovian.commit(project='first-steps-with-python')