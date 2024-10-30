import numpy as np

np.random.seed(1)
A = np.random.uniform(size=[3,3])
B = np.random.uniform(size=[3,3])
q = np.random.uniform(size=[3,1])

print(np.matmul(A, np.matmul(B,q)))
print(np.matmul(np.matmul(B, A.T),q))