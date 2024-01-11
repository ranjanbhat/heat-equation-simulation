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
N = 200  # Number of grid points
T = 0.1  # Simulation time
L = 1.0  # Domain length
dt = 0.00001  # Time step
alpha = 1  # Diffusion coefficient
num_steps = int(T / dt)  # Number of time steps

# Read data from the main output file
data = read_data('result.txt')
fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
X, T_vals = np.meshgrid(np.linspace(0, L, N, endpoint=False), np.linspace(0, T, num_steps))
ax.plot_surface(X, T_vals, data, cmap='viridis')
ax.set_title(f"Heat Equation Solution")
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u')

true = np.array([np.exp(-4 * np.pi * np.pi * alpha * step * dt) * np.sin(2 * np.pi * np.linspace(0, L, N, endpoint=False)) for step in range(num_steps)])
errors = np.sum(np.abs(data - true), axis=1) / N

# errors = read_data('error.txt')
fig2 = plt.figure(2)
plt.plot(np.linspace(0, T, num_steps), errors)
plt.xlabel('t')
plt.ylabel('E(t)')
plt.title(f"Error")

plot_time_steps = [0, 100, 500, 1000]
plt.figure(3)
x = np.linspace(0, L, N, endpoint=False)
for step in plot_time_steps:
    step_data = data[step]
    plt.plot(x, step_data.reshape(x.shape), label=f"t = {step * dt}")
plt.legend()
plt.xlabel('x')
plt.ylabel('u')
plt.title(f"Heat Equation Solution")


plt.show()