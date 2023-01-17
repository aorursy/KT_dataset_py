import numpy as np

from sklearn import datasets, linear_model



x = np.array([[2, 0], [3, -1], [4,5]]) # Define x as np.array([[2, 0], [3, -1], [4,5]])

y = np.array([3, 2, 1]) # Define y as np.array([3, 2, 1])

classifier = linear_model.SGDClassifier() # Declare a linear classifier that uses SGD to minimize

# its loss. Define it as linear_model.SGDClassifier()

classifier.fit(x, y)



#We're going to classify sets of two numbers.

# Adjust the step size.

# Adjut the max_iters.



# Now let's try your classifier on some new data!

print(classifier.predict([[0.2, -1]]))



# http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html