from __future__ import absolute_import, division, print_function
import sys
print("Python version:", sys.version)

import tensorflow as tf
print("TensorFlow version:", tf.__version__)
# TEMPORARY: enable control flow for @tf.function decorators
from tensorflow.python.ops import control_flow_util
control_flow_util.ENABLE_CONTROL_FLOW_V2 = True
@tf.function
def print_1(should_print):
    print('print_1:', should_print)
    assert should_print == True
    for x in range(10):
        if x >= 5:
            print(x)
            for y in range (10):
                if y < 5:
                    print('\t', y)
print(tf.autograph.to_code(print_1.python_function,  experimental_optional_features=None))
# check with correct condition
print_1(True)
# check without passing value which will give us TypeError
try:
    print_1()
except TypeError as e:
    print('TypeError as expected:', e)
# Now let's add a default value so that we don't get a TypeError
@tf.function
def print_2(should_print=False):
    print('print_2:', should_print)
    assert should_print == True
    for x in range(10):
        if x >= 5:
            print(x)
            for y in range (10):
                if y < 5:
                    print('\t', y)

print(tf.autograph.to_code(print_2.python_function,  experimental_optional_features=None))
print_2(True)
try:
    print_2()
except AssertionError as e:
    print('Assertion Error as expected:', e)
# Let's try catching AssertionError in our function
@tf.function
def print_3(should_print=False):
    try:
        assert should_print == False
        print('print_3:', should_print)
        for x in range(10):
            if x >= 5:
                print(x)
                for y in range (10):
                    if y < 5:
                        print('\t', y)
    except AssertionError as e:
        print('Should print assertion failed', e)
try:
    print(tf.autograph.to_code(print_3.python_function,  experimental_optional_features=None))
except NotImplementedError as e:
    print('NotImplementedError as expected:', e)
# let's try with reverting our assertion and not passing any value
@tf.function
def print_4(should_print=False):
    assert should_print == False
    print('print_4:', should_print)
    for x in range(10):
        if x >= 5:
            print(x)
            for y in range (10):
                if y < 5:
                    print('\t', y)

print(tf.autograph.to_code(print_4.python_function,  experimental_optional_features=None))
print_4()
# works as expected, even though it seems like the converted code set the default value to None