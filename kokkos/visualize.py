import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Function to read data from the C++ output files
def read_data(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            values = [float(val) for val in line.split()]
            data.append(values)
    return np.array(data)

# Parameters
N = int(os.sys.argv[1]) # Number of grid points
T = 0.1  # Simulation time
L = 1.0  # Domain length
alpha = 1  # Diffusion coefficient
sigma = 0.15 #Courant number
dx = L / N  # Grid spacing
dt = sigma * dx * dx / alpha  # Time step size
num_steps = int(T / dt)  # Number of time steps

# # Read data from the main output file
data = read_data(f'results_{N}.txt')
# fig = plt.figure(1)
# ax = fig.add_subplot(111, projection='3d')
# X, T_vals = np.meshgrid(np.linspace(0, L, N, endpoint=False), np.linspace(0, T, num_steps))
# ax.plot_surface(X, T_vals, data, cmap='viridis')
# ax.set_title(f"Heat Equation Solution")
# ax.set_xlabel('x')
# ax.set_ylabel('t')
# ax.set_zlabel('u')

# # print data from a particular time step and location
# print(f"u({dt*1000}, {L/4}) = {data[1000][N//4]}")
# print(f"sin(2*pi*L/4) = {np.sin(2 * np.pi * L / 4)}")
# c = np.log( data[1000][N//4] / np.sin(2 * np.pi * L / 4) ) / (-4 * np.pi * np.pi * alpha * dt * 1000)
# print(f"c = {c}")

print(f"num_steps = {num_steps}")

true = np.array([np.exp(-4 * np.pi * np.pi * alpha * step * dt) * np.sin(2 * np.pi * np.arange(0, L, dx)) for step in range(num_steps)])
# errors = np.sum(np.abs(data - true), axis=1) / N
# print(errors[0])
errors = read_data('error_with_time.txt')

# errors = read_data('error.txt')
fig2 = plt.figure(2)
plt.plot(np.arange(0, T-dt, dt), errors)
plt.xlabel('t')
plt.ylabel('E(t)')
plt.title(f"Error")

plot_time_steps = [0, 0.05, 0.1, 0.25, 0.5]
plot_time_steps = [int(num_steps*step) for step in plot_time_steps]
plt.figure(3)
x = np.arange(0, L, dx)
for step in plot_time_steps:
    step_data = data[step]  
    plt.plot(x, step_data.reshape(x.shape), label=f"t = {step * dt}")
    # in dashed lines
    plt.plot(x, true[step], '--', label=f"t = {step * dt} (true)")
plt.legend()
plt.xlabel('x')
plt.ylabel('u')
plt.title(f"Heat Equation Solution")
# save figure 3
plt.savefig(f"comparison_plots.png")

# true = np.array([np.exp(-4 * np.pi * np.pi * 4 *  alpha * step * dt) * np.sin(2 * np.pi * np.linspace(0, L, N, endpoint=False)) for step in range(num_steps)])

# fig = plt.figure(4)
# ax = fig.add_subplot(111, projection='3d')
# X, T_vals = np.meshgrid(np.linspace(0, L, N, endpoint=False), np.linspace(0, T, num_steps))
# ax.plot_surface(X, T_vals, true, cmap='viridis')
# ax.set_title(f"True Solution")
# ax.set_xlabel('x')
# ax.set_ylabel('t')
# ax.set_zlabel('u')

plt.show()