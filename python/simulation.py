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
N = 50         # Number of grid points
dx = L / N      # Grid spacing
dt = 0.00001    # Time step
num_steps = int(T / dt)    # Number of time steps

plot_time_steps = [0, 100, 500, 1000]

filename = f"u_3d_{alpha}_{dx}_{dt}_{N}_{num_steps}"

def display_and_save_plots(u_3d, filename):
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')
    X, T_vals = np.meshgrid(np.linspace(0, L, N, endpoint=False), np.linspace(0, T, num_steps))
    ax.plot_surface(X, T_vals, u_3d, cmap='viridis')
    ax.set_title(f"Heat Equation Solution")
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('u')
    plt.savefig(f'{filename}_3d.png')

    plt.figure(2)
    x = np.linspace(0, L, N, endpoint=False)
    for step in plot_time_steps:
        plt.plot(x, u_3d[step], label=f"t = {step * dt}")
        solution = np.exp(-4 * np.pi * np.pi * alpha * step * dt) * np.sin(2 * np.pi * x)
        # mean_error = np.sum(np.abs(u_3d[step] - solution)) / N
        # print(f"Mean error at t = {step * dt} is {mean_error}")
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('u')
    plt.title(f"Heat Equation Solution")
    plt.savefig(f'{filename}_2d.png')
    print_plot_errors(u_3d, filename)
    plt.show()

def print_plot_errors(u_3d, filename):
    errors = []
    x = np.linspace(0, L, N, endpoint=False)
    for step in range(u_3d.shape[0]):
        solution = np.exp(-4 * np.pi * np.pi * alpha * step * dt) * np.sin(2 * np.pi * x)
        mean_error = np.sum(np.abs(u_3d[step] - solution)) / N
        print(f"E({step * dt}) = {mean_error}")
        errors.append(mean_error)
    fig = plt.figure(3)
    plt.plot(np.linspace(0, T, num_steps), errors)
    plt.xlabel('t')
    plt.ylabel('E(t)')
    plt.savefig(f'{filename}_error.png')
    plt.title(f"Error")

def main():
    if os.path.exists(f'{filename}.npy'):
        u_3d = np.load(f'{filename}.npy')
        print("Loaded data from file")
        display_and_save_plots(u_3d, filename)
        exit()

    x = np.linspace(0, L, N, endpoint=False)
    u = initial_condition(x)
    u_3d = np.zeros((num_steps, N))

    for step in range(num_steps):
        u_3d[step] = u
        new_u = np.zeros(N)

        for i in range(N):
            left = (i - 1 + N) % N
            right = (i + 1) % N
            new_u[i] = u[i] + alpha * dt / (dx * dx) * (u[left] - 2 * u[i] + u[right])
            print(new_u[i] - u[i])

        u = new_u

    display_and_save_plots(u_3d, filename)
    np.save(f'{filename}.npy', u_3d)

if __name__ == "__main__":
    main()
