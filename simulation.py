import numpy as np
import os
import matplotlib.pyplot as plt

# Parameters
L = 1.0         # Domain length
T = 0.1        # Simulation time
alpha = 1       # Diffusion coefficient

# Initial condition function
def initial_condition(x):
    return np.sin(2 * np.pi * x)

# Grid and time step
N = 128         # Number of grid points
dx = L / N      # Grid spacing
dt = 0.00001    # Time step
num_steps = int(T/dt)    # Number of time steps

plot_time_steps = [0, 100, 500, 1000]

filename = f"u_3d_{alpha}_{dx}_{dt}_{N}_{num_steps}"

if os.path.exists(f'{filename}.npy'):
    u_3d = np.load(f'{filename}.npy')
    print("Loaded data from file")
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')
    X, T = np.meshgrid(np.linspace(0, L, N, endpoint=False), np.linspace(0, T, num_steps))
    ax.plot_surface(X, T, u_3d, cmap='viridis')
    ax.set_title(f"Diffusion Equation Solution")
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('u')
    plt.savefig(f'{filename}_3d.png')
    plt.show()
    plt.figure(2)
    x = np.linspace(0, L, N, endpoint=False)
    for step in plot_time_steps:
        plt.plot(x, u_3d[step], label=f"t = {step * dt}")
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('u')
    plt.title(f"Diffusion Equation Solution")
    plt.savefig(f'{filename}_2d.png')
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

    u = new_u

# Create 3D plot
fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, T, u_3d, cmap='viridis')

# save figure and data, file name must include alpha, dx, dt, N, num_steps
np.save(f'{filename}.npy', u_3d)
plt.savefig(f'{filename}_3d.png')

ax.set_title(f"Diffusion Equation Solution")
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u')
plt.show()

# Create 2D plot
plt.figure(2)
for step in plot_time_steps:
    plt.plot(x, u_3d[step], label=f"t = {step * dt}")
    # Calculate average error at these time steps
    solution = np.exp(-4 * np.pi * np.pi * alpha * step * dt) * np.sin(2 * np.pi * x)
    mean_error = np.sum(np.abs(u_3d[step] - solution)) / N
    rms_error = np.sqrt(np.sum((u_3d[step] - solution) ** 2) / N)
    print(f"Mean error at t = {step * dt} is {mean_error}")
    print(f"RMS error at t = {step * dt} is {rms_error}")

plt.xlabel('x')
plt.ylabel('u')
plt.title(f"Diffusion Equation Solution")
plt.legend()
np.save(f'{filename}.npy', u_3d)
plt.savefig(f'{filename}_2d.png')
plt.show()


