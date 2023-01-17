import numpy as np
eps = 1.0e-6

# cosine similarity

# eps: avoid dividing by 0

def cosSim(inA, inB):

    denom = np.linalg.norm(inA) * np.linalg.norm(inB)

    return float(inA * inB.T) / (denom + eps)
# Matrix of user to item

# row: user, column: item

A = np.mat([[5, 5, 3, 0, 5, 5], [5, 0, 4, 0, 4, 4], [0, 3, 0, 5, 4, 5], [5, 4, 3, 3, 5, 5]])
print('A is {} by {} matrix: '.format(A.shape[0], A.shape[1]))

print(A)
U, S, VT = np.linalg.svd(A.T)

V = VT.T

Sigma = np.diag(S)

print('AT shape = ', A.T.shape)

print('U shape = ', U.shape)

print('VT shape = ', VT.shape)

print('S = ', S)

print('Sigma = \n', Sigma)
# Use first 2 singular values

r = 2

# Get approximate U, Sigma, VT

Ur = U[:, :r]

Sr = Sigma[:r, :r]

Vr = V[:, :r]

print('Ur(matrix of item to latent factor), shape = ', Ur.shape)

print('Sr(matrix of singular values), shape = ', Sr.shape)

print('Vr(matrix of user to latent factor), shape = ', Vr.shape)
# Vector of new user to item

new = np.mat([[5, 5, 0, 0, 0, 5]])

newresult = new * Ur * np.linalg.inv(Sr)

print('Vector of new user to latent factor = ', newresult)
cos_sim = np.array([cosSim(newresult, vi) for vi in Vr])

recommended_user_id = np.argsort(-cos_sim)[0]

print('Most similar user id = ', recommended_user_id)

print('Cosine similarity = ', cos_sim[0])
recommended_item_id = []

for idx, item in enumerate(np.ndarray.flatten(np.array(new))):

    if item == 0 and A[recommended_user_id, idx] != 0:

        recommended_item_id.append(idx)

print('Recommended item id = ', recommended_item_id)