import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

A = np.matrix([[1, 1, 1, 1], [1, -1, 1, 1], [1, 1, -1, 1], [-1, -1, -1, 1], [1, 1, 1, -1]])
print("Matriz A: ")
print(A)

# 1.A
U, s, V = np.linalg.svd(A)
U = U[:, :-1]
S = np.diag(s)

print("SVD Reducida:")
print ("U:")
print(U)
print ("S:")
print(S)
print ("V:")
print(V)

# 1.B
U, s, V = np.linalg.svd(A)
S = np.zeros((5, 4), dtype=complex)
S[:4, :4] = np.diag(s)

print("SVD Completa:")
print ("U:")
print(U)
print ("S:")
print(S)
print ("V:")
print(V)

# Data Set Elipse
dataSetElipse = open("Data/data_set_elipse.txt", "r")
x = []
y = []
z = []
for line in dataSetElipse:
	data = line.split(" ")
	x.append((float)(data[0]))
	y.append((float)(data[1]))
	z.append((float)(data[2]))

# 1.C
plt.plot(x, y, 'ro')
plt.axis([-10, 10, -10, 10])
plt.show()

matrix = []
for i in range(0, len(x)):
	matrix.append([(float)(x[i]), (float)(y[i])])

A = np.matrix(matrix)
U, s, V = np.linalg.svd(A, full_matrices=False)

v1_x = [0, V[0,0]]
v1_y = [0, V[0,1]]
v2_x = [0, V[1,0]]
v2_y = [0, V[1, 1]]

plt.plot(x, y, 'ro')
plt.plot(v1_x, v1_y)
plt.plot(v2_x, v2_y)
plt.axis([-7, 7, -7, 7])
plt.show()

# 1.D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c = 'r', marker = 'o')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()

matrix = []
for i in range(0, len(x)):
	matrix.append([(float)(x[i]), (float)(y[i]), (float)(z[i])])

A = np.matrix(matrix)
U, s, V = np.linalg.svd(A, full_matrices=False)

v1_x = [0, V[0,0]]
v1_y = [0, V[0,1]]
v1_z = [0, V[0,2]]
v2_x = [0, V[1,0]]
v2_y = [0, V[1, 1]]
v2_z = [0, V[1, 2]]
v3_x = [0, V[2,0]]
v3_y = [0, V[2, 1]]
v3_z = [0, V[2, 2]]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c = 'r', marker = 'o')
ax.plot(v1_x, v1_y, v1_z, c='b')
ax.plot(v2_x, v2_y, v2_z, c='g')
ax.plot(v3_x, v3_y, v3_z, c='y')
plt.show()

# 1.E
dataSetX = open("Data/data_set_X.txt", 'r')
X = []
for line in dataSetX:
	X_s = []
	for value in line.split(" "):
		X_s.append((float)(value))
	X.append(X_s)

U, s, V = np.linalg.svd(X, full_matrices=False)
S = np.diag(s)

C = np.matrix(U) * np.matrix(S)
D = V

plotx = []
ploty = []

for m in range(0, 101):
	X_m = np.matrix(C[:, 0:m]) * np.matrix(D[0:m, :])
	det = np.matrix(X) - X_m
	ploty.append(np.linalg.norm(det))
	plotx.append(m)

plt.plot(plotx, ploty)
plt.ylabel('||X - X_m||')
plt.xlabel('m')
plt.show()