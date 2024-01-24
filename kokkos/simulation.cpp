#include <iostream>
#include <fstream>
#include <Kokkos_Core.hpp>

// Parameters
constexpr double L = 1.0;         // Domain length
constexpr double T = 0.1;         // Simulation time
constexpr double alpha = 1.0;      // Diffusion coefficient
constexpr double sigma = 0.15;     // CFL stability number

// Initial condition function
double initial_condition(double x) {
    return std::sin(2.0 * M_PI * x);
}

int main(int argc, char* argv[]) {
    // Grid and time step
    const int N = static_cast<int>(std::stoi(argv[1]));  // Number of grid points
    const double dx = L / N;         // Grid spacing
    const double dt = sigma * dx*dx / alpha;  // Time step
    const int num_steps = static_cast<int>(T / dt);  // Number of time steps
    double error_total = 0.0;

    // Kokkos initialization
    Kokkos::initialize();

    {

        // Define views for Kokkos
        Kokkos::View<double*> u("u", N);
        Kokkos::View<double**> u_3d("u_3d", num_steps, N);
        Kokkos::View<double*> errors("errors", num_steps);
        Kokkos::View<double*> u_frac("u_frac", N);

        // Initialize initial condition
        Kokkos::parallel_for("Initialize", N, KOKKOS_LAMBDA(const int i) {
            double x = i * dx;
            u(i) = initial_condition(x);
        });

        // copy u to u_frac
        Kokkos::parallel_for("Copy", N, KOKKOS_LAMBDA(const int i) {
            u_frac(i) = u(i);
        });

        // Time-stepping loop
        for (int step = 0; step < num_steps; ++step) {

            // Copy data to u_3d
            Kokkos::parallel_for("CopyToResult", N, KOKKOS_LAMBDA(const int i) {
                u_3d(step, i) = u(i);
            });

            double delta[4] = {0, 1, 1.0/3, 1};
            double a[3] = {0, -4.0, 1.0/27};
            double b[3] = {1, 2.0/9, 3.0/4};

            Kokkos::View<double*> q("q", N);

            // Kokkos::fence();

            for (int m = 0; m < 3; m++) {

                Kokkos::View<double*> u_temp("u_temp", N);

                Kokkos::parallel_for("Diffusion", N, KOKKOS_LAMBDA(const int i) {
                    int left1 = (i - 1 + N) % N;
                    int left2 = (i - 2 + N) % N;
                    int left3 = (i - 3 + N) % N;
                    int right1 = (i + 1) % N;
                    int right2 = (i + 2) % N;
                    int right3 = (i + 3) % N;

                    q(i) = a[m]*q(i) + alpha * (2*u_frac(left3) - 27*u_frac(left2) + 270*u_frac(left1) - 490*u_frac(i) + 270*u_frac(right1) - 27*u_frac(right2) + 2*u_frac(right3)) / (180*dx*dx) * dt;
                    // qi = a[m]*qi + alpha * (u_frac(left1) - 2*u_frac(i) + u_frac(right1)) / (dx*dx) * dt;
                    u_temp(i) = u_frac(i) + b[m]*q(i);

                });

                Kokkos::fence();

                // copy u_temp to u_frac
                Kokkos::parallel_for("Copy", N, KOKKOS_LAMBDA(const int i) {
                    u_frac(i) = u_temp(i);
                });

                Kokkos::fence();

            }

            // copy u_frac to u
            Kokkos::parallel_for("Copy", N, KOKKOS_LAMBDA(const int i) {
                u(i) = u_frac(i);
            });

            Kokkos::fence();

            // Calculate error
            double error = 0.0;
            Kokkos::parallel_reduce("Error", N, KOKKOS_LAMBDA(const int i, double& error) {
                double x = i * dx;
                double diff = std::abs(u(i) - std::exp(-4.0 * M_PI * M_PI * alpha * step * dt) * std::sin(2.0 * M_PI * x));
                error += diff;
            }, error);
            errors(step) = error/N;
            error_total += error/N;

            // // Note: You may want to synchronize Kokkos view data if needed
            // Kokkos::fence();
        }

        // Write results to a file results_N.txt
        // std::ofstream outFile("results_" + std::to_string(N) + ".txt");
        // for (int step = 0; step < num_steps; ++step) {
        //     for (int i = 0; i < N; ++i) {
        //         outFile << u_3d(step, i) << " ";
        //     }
        //     outFile << "\n";
        // }
        // outFile.close();

        // Write errors to a file
        // std::ofstream outFile2("error.txt");
        // for (int step = 0; step < num_steps; ++step) {
        //     outFile2 << errors(step) << "\n";
        // }
        // outFile2.close();

        // Append error into a file
        std::ofstream outFile3("error.txt", std::ios_base::app);
        outFile3 << N << " "<< error_total/num_steps << "\n";
        outFile3.close();
    }

    // Kokkos finalization
    Kokkos::finalize();

    // print Complete!
    std::cout << "Complete!" << std::endl;

    return 0;
}
