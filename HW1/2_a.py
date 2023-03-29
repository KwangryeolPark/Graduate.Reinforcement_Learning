# %%
import numpy

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

p = numpy.linalg.inv(
    numpy.eye(50) - P
)

print(f'Stationary distribution p:\n{p}')