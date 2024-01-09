#include <iostream>
#include <fstream>
#include <Kokkos_Core.hpp>

// Parameters
constexpr double L = 1.0;         // Domain length
constexpr double T = 0.1;         // Simulation time
constexpr double alpha = 1.0;      // Diffusion coefficient

// Initial condition function
double initial_condition(double x) {
    return std::sin(2.0 * M_PI * x);
}

int main() {
    // Grid and time step
    constexpr int N = 50;            // Number of grid points
    const double dx = L / N;         // Grid spacing
    const double dt = 0.00001;       // Time step
    const int num_steps = static_cast<int>(T / dt);  // Number of time steps

    // Kokkos initialization
    Kokkos::initialize();

    {

        // Define views for Kokkos
        Kokkos::View<double*> u("u", N);
        Kokkos::View<double**> u_3d("u_3d", num_steps, N);

        // Initialize initial condition
        Kokkos::parallel_for("Initialize", N, KOKKOS_LAMBDA(const int i) {
            double x = i * dx;
            u(i) = initial_condition(x);
        });

        // Time-stepping loop
        for (int step = 0; step < num_steps; ++step) {

            // Update u
            Kokkos::parallel_for("Diffusion", N, KOKKOS_LAMBDA(const int i) {
                int left = (i - 1 + N) % N;
                int right = (i + 1) % N;
                double new_u = u(i) + alpha * dt / (dx * dx) * (u(left) - 2.0 * u(i) + u(right));
                u(i) = new_u;
            });

            // Copy data to u_3d
            Kokkos::parallel_for("CopyToResult", N, KOKKOS_LAMBDA(const int i) {
                u_3d(step, i) = u(i);
            });

            // Note: You may want to synchronize Kokkos view data if needed
            Kokkos::fence();
        }

        // Write results to a file
        std::ofstream outFile("result.txt");
        for (int step = 0; step < num_steps; ++step) {
            for (int i = 0; i < N; ++i) {
                outFile << u_3d(step, i) << " ";
            }
            outFile << "\n";
        }
        outFile.close();


    }

    // Kokkos finalization
    Kokkos::finalize();

    return 0;
}
