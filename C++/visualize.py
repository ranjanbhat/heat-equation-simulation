import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function to read data from the C++ output files
def read_data(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            values = [float(val) for val in line.split()]
            data.append(values)
    return np.array(data)

# Parameters
N = 128  # Number of grid points
T = 0.1  # Simulation time
L = 1.0  # Domain length
dt = 0.00001  # Time step
alpha = 1  # Diffusion coefficient
num_steps = int(T / dt)  # Number of time steps

# Read data from the main output file
data = read_data('u_data.txt')

fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
X, T_vals = np.meshgrid(np.linspace(0, L, N, endpoint=False), np.linspace(0, T, num_steps))
ax.plot_surface(X, T_vals, data, cmap='viridis')
ax.set_title(f"Heat Equation Solution")
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u')

plot_time_steps = [0, 100, 500, 1000]
plt.figure(2)
x = np.linspace(0, L, N, endpoint=False)
for step in plot_time_steps:
    data = read_data(f'u_data_{step}.txt')
    plt.plot(x, data.reshape(x.shape), label=f"t = {step * dt}")
    solution = np.exp(-4 * np.pi * np.pi * alpha * step * dt) * np.sin(2 * np.pi * x)
plt.legend()
plt.xlabel('x')
plt.ylabel('u')
plt.title(f"Heat Equation Solution")

plt.show()