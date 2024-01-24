import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Parameters
N_vals = [64, 128, 256, 512, 1024] # Number of grid points
T = 0.1  # Simulation time
L = 1.0  # Domain length
alpha = 1  # Diffusion coefficient
# sigma = 0.15 #Courant number
# dt = sigma * (L / N) * (L / N) / alpha  # Time step size
# num_steps = int(T / dt)  # Number of time steps
N_vals_calculated = []

# Read data from the main output file if it exists
if os.path.exists(f"error.txt"):
    with open(f"error.txt", 'r') as file:
        for line in file:
            N_vals_calculated.append(int(line.split()[0]))
        
# for N_vals that have not been calculated, run the simulation
for N in N_vals:
    if N in N_vals_calculated:
        continue
    os.system(f"./simulation.host {N}")

# Read data from the main output file
errors = []
with open(f"error.txt", 'r') as file:
    for line in file:
        error = float(line.split()[1])
        errors.append(error)
    
print(errors)
print(N_vals)
# print slope
print(np.log(errors[1] / errors[0]) / np.log(N_vals[1] / N_vals[0]))

# plot log errors vs log N
plt.figure(1)
plt.loglog(N_vals, errors)
plt.xlabel('N')
plt.ylabel('E')
plt.title(f"Error")
# plt.legend([f"t = {step * dt}" for step in plot_time_steps])
plt.savefig("order_of_accuracy.png")
plt.show()

