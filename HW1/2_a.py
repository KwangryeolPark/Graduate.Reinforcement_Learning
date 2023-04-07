# %%
import numpy
import matplotlib.pyplot as plt

size = 50

P = numpy.zeros((size, size))

def set_P(pos1, pos2, value):
    global P
    P[pos1 - 1][pos2 - 1] = value

set_P(1, 1, 1/2)
set_P(2, 1, 1/2)
set_P(50, 50, 1/2)
set_P(49, 50, 1/2)

for i in range(2, 50):
    set_P(i - 1, i, 1/3)
    set_P(i, i, 1/3)
    set_P(i + 1, i, 1/3)
# Solution for p = Pp -> p is the eigen vector whose eigen value is 1.
P_eigval, P_eigvec = numpy.linalg.eig(P)
P_eigval_1_index = numpy.argmin(1 - P_eigval)   # Seek the index of eigenval == 1 (or almost 1 because of floating err)
# p is normalized vector
p = P_eigvec[:, P_eigval_1_index]
# so the total sum of values is not 1 -> make it one.
total = numpy.sum(p)
p = p / total
plt.plot(p)

print(f'Stationary distribution p:\n{p}')