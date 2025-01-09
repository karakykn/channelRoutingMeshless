import numpy as np

np.random.seed(3)
x = np.random.random(3)
Psi_x = np.random.random((3,3))
PsiInv = np.random.random((3,3))

ans1 = np.matmul(Psi_x, np.matmul(PsiInv, x))
print(ans1)

ans2 = np.matmul(np.matmul(PsiInv, Psi_x.T), x)
print(ans2)