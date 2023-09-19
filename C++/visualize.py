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
dt = 0.00001  # Time step
num_steps = int(T / dt)  # Number of time steps

# Read data from the main output file
data = read_data('u_data.txt')

# Create a 3D plot for the full domain
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = np.linspace(0, 1, N, endpoint=False)
t = np.linspace(0, T, num_steps)
X, T = np.meshgrid(x, t)
ax.plot_surface(X, T, data, cmap='viridis')
ax.set_title('Heat Equation Solution (3D)')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u')
plt.show()

# Read data from specific time step files and create 2D plots
plot_time_steps = [0, 100, 500, 1000]
for step in plot_time_steps:
    data_step = read_data(f'u_data_{step}.txt')
    plt.figure()
    plt.plot(x, data_step, label=f't = {step * dt}')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.legend()
    plt.title('Heat Equation Solution (2D)')
    plt.show()
