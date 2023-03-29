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

p = numpy.zeros((size))
p[0] = 1
for i in range(5000):
    if i % 500 == 0:
        plt.figure(str(i))
        plt.plot(range(size), p)
        plt.show()
        plt.close()
    p = numpy.matmul(P, p)


# Compare
p = numpy.linalg.inv(
    numpy.eye(50) - P
)
plt.figure()
plt.plot(range(size), p)
plt.show()
plt.close()

# 처음에는 상태 1에 확률이 1이었다가, 점점 모든 영역에 걸쳐 유사한 확률 값을 갖는 분포로 변함.
# %%
# import imageio
# import os
# images = []

# for index, file in enumerate(os.listdir('.')):
#     print(index, end='\r')
#     if 'jpg' in file and index % 50 == 0:        
#         images.append(imageio.imread(file))
        
# imageio.mimsave('./2_b.gif', images)        