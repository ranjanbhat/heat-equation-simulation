import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters
L = 1.0         # Domain length
T = 0.1        # Simulation time
alpha = 1       # Diffusion coefficient

# Initial condition function
def initial_condition(x):
    return np.sin(2 * np.pi * x)

# Grid and time step
N = 100         # Number of grid points
dx = L / N      # Grid spacing
dt = 0.00001    # Time step
num_steps = int(T/dt)    # Number of time steps

filename = f"u_3d_{alpha}_{dx}_{dt}_{N}_{num_steps}"

if os.path.exists(f'{filename}.npy'):
    u_3d = np.load(f'{filename}.npy')
    print("Loaded data from file")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, T = np.meshgrid(np.linspace(0, L, N, endpoint=False), np.linspace(0, T, num_steps))
    ax.plot_surface(X, T, u_3d, cmap='viridis')
    ax.set_title(f"Diffusion Equation Solution")
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('u')
    plt.show()
    exit()

# Initialize grid and solution arrays
x = np.linspace(0, L, N, endpoint=False)
u = initial_condition(x)

# Create meshgrid for 3D plotting
X, T = np.meshgrid(x, np.linspace(0, T, num_steps))

# Initialize the 3D array to store the solution
u_3d = np.zeros((num_steps, N))

# Perform time-stepping and store solutions
for step in range(num_steps):
    u_3d[step] = u
    
    new_u = np.zeros(N)

    for i in range(N):
        left = (i - 1 + N) % N    # Periodic boundary conditions
        right = (i + 1) % N

        new_u[i] = u[i] + alpha * dt / (dx * dx) * (u[left] - 2 * u[i] + u[right])
        print(new_u[i] - u[i])

        # Dirichlet boundary conditions
        # if i == 0 or i == N - 1:
        #     new_u[i] = 0
        # else:
        #     new_u[i] = u[i] + alpha * dt / (dx * dx) * (u[i - 1] - 2 * u[i] + u[i + 1])

    u = new_u

# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, T, u_3d, cmap='viridis')

# save figure and data, file name must include alpha, dx, dt, N, num_steps
np.save(f'{filename}.npy', u_3d)
plt.savefig(f'{filename}.png')

ax.set_title(f"Diffusion Equation Solution")
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u')

plt.show()
