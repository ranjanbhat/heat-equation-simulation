#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>

// Parameters
const double L = 1.0;           // Domain length
const double T = 0.1;           // Simulation time
const double alpha = 1.0;       // Diffusion coefficient

// Initial condition function
double initial_condition(double x) {
    return sin(2 * M_PI * x);
}

int main() {
    // Grid and time step
    const int N = 128;                   // Number of grid points
    const double dx = L / N;             // Grid spacing
    const double dt = 0.00001;           // Time step
    const int num_steps = T / dt;        // Number of time steps

    std::vector<double> u(N);
    std::vector<double> new_u(N);

    // Initialize the initial condition
    for (int i = 0; i < N; i++) {
        double x = i * dx;
        u[i] = initial_condition(x);
    }

    std::ofstream outputFile("u_data.txt");  // Open a file for saving all data

    // Time-stepping loop
    for (int step = 0; step < num_steps; step++) {
        for (int i = 0; i < N; i++) {
            int left = (i - 1 + N) % N;
            int right = (i + 1) % N;
            new_u[i] = u[i] + alpha * dt / (dx * dx) * (u[left] - 2 * u[i] + u[right]);
        }

        // Calculate and output the average error in the domain
        double total_error = 0.0;
        for (int i = 0; i < N; i++) {
            double x = i * dx;
            double analytical_solution = sin(2 * M_PI * x) * exp(-4 * M_PI * M_PI * alpha * (step * dt));
            total_error += std::abs(u[i] - analytical_solution);
        }
        double mean_error = total_error / N;

        std::cout << "Mean error at t = " << step * dt << " is " << mean_error << std::endl;

        // Save the current state of 'u' into the main data file
        for (int i = 0; i < N; i++) {
            outputFile << u[i] << " ";
        }
        outputFile << "\n";

        // Update u for the next time step
        u = new_u;
    }

    outputFile.close();  // Close the main data file

    return 0;
}
