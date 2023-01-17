import tensorflow as tf



tf.enable_eager_execution()
x = tf.ones((2, 2))

with tf.GradientTape() as t:

    t.watch(x)

    y = tf.reduce_sum(x)

    z = tf.multiply(y, y)

    print(z)



# Derivative of z with respect to the original input tensor x

dz_dx = t.gradient(z, x)

print(dz_dx)
x = tf.ones((2, 2))

  

with tf.GradientTape() as t:

  t.watch(x)

  y = tf.reduce_sum(x)

  z = tf.multiply(y, y)



# Use the tape to compute the derivative of z with respect to the

# intermediate value y.

dz_dy = t.gradient(z, y)

print(dz_dy.numpy())
x = tf.constant([[3.0,5.0],[2.0,8.0]])

print(x)

with tf.GradientTape(persistent=True) as t:

  t.watch(x)

  y = x * x

  print(y)

  z = y * y

dz_dx = t.gradient(z, x)  # 108.0 (4*x^3 at x = 3)

dy_dx = t.gradient(y, x)  # 6.0

print(dz_dx)

print(dy_dx)

del t  # Drop the reference to the tape
def f(x, y):

  output = 1.0

  for i in range(y):

    if i > 1 and i < 5:

      output = tf.multiply(output, x)

  return output



def grad(x, y):

  with tf.GradientTape() as t:

    t.watch(x)

    out = f(x, y)

  return t.gradient(out, x) 



x = tf.convert_to_tensor(2.0)



print(grad(x, 6).numpy())

print(grad(x, 5).numpy())

print(grad(x, 4).numpy())
x = tf.Variable(1.0)  # Create a Tensorflow variable initialized to 1.0 

# note tf.Variable created a Trainable=true variable thats automatically watched



with tf.GradientTape() as t:

  with tf.GradientTape() as t2:

    y = x * x * x

  # Compute the gradient inside the 't' context manager

  # which means the gradient computation is differentiable as well.

  dy_dx = t2.gradient(y, x)

d2y_dx2 = t.gradient(dy_dx, x)



print(dy_dx.numpy())

print(d2y_dx2.numpy())