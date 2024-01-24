#include <iostream>
#include <fstream>
#include <Kokkos_Core.hpp>

// Parameters
constexpr double L = 1.0;         // Domain length
constexpr double T = 0.1;         // Simulation time
constexpr double alpha = 1.0;      // Diffusion coefficient
constexpr double sigma = 0.15;     // CFL stability number
constexpr int Nbuf = 3;            // Number of buffer cells

// Initial condition function
double initial_condition(double x) {
    return std::sin(2.0 * M_PI * x);
}

// function to update ghost cells, pass by reference
void update_ghost_cells(Kokkos::View<double*> u, const int N) {
    // Update ghost cells
    Kokkos::parallel_for("UpdateGhostCells", Nbuf, KOKKOS_LAMBDA(const int i) {
        u(i) = u(i + N);
        u(i + N + Nbuf) = u(i + Nbuf);
    });
}

int main(int argc, char* argv[]) {
    // Grid and time step
    const int N = static_cast<int>(std::stoi(argv[1]));  // Number of grid points
    const double dx = L / N;         // Grid spacing
    const double dt = sigma * dx*dx / alpha;  // Time step
    const int num_steps = static_cast<int>(T / dt);  // Number of time steps
    double error_total = 0.0;

    int time_steps_to_save[5] = {static_cast<int>(num_steps*0), 
                                 static_cast<int>(num_steps*0.05), 
                                 static_cast<int>(num_steps*0.1), 
                                 static_cast<int>(num_steps*0.25), 
                                 static_cast<int>(num_steps*0.5)};
    // Kokkos initialization
    Kokkos::initialize();

    {

        // Define views for Kokkos
        Kokkos::View<double*> u("u", N + 2*Nbuf);
        // Kokkos::View<double**> u_3d("u_3d", num_steps, N);
        Kokkos::View<double*> errors("errors", num_steps);
        Kokkos::View<double*> u_frac("u_frac", N + 2*Nbuf);

        // Initialize initial condition
        Kokkos::parallel_for("Initialize", N, KOKKOS_LAMBDA(const int i) {
            double x = i * dx;
            u(i + Nbuf) = initial_condition(x);
        });

        // Update ghost cells
        update_ghost_cells(u, N);

        // copy u to u_frac
        Kokkos::parallel_for("Copy", N+2*Nbuf, KOKKOS_LAMBDA(const int i) {
            u_frac(i) = u(i);
        });

        // Time-stepping loop
        for (int step = 0; step < num_steps; ++step) {

            // Copy data to u_3d
            // Kokkos::parallel_for("CopyToResult", N, KOKKOS_LAMBDA(const int i) {
            //     u_3d(step, i) = u(i + Nbuf);
            // });
        
            // Write results to a file results_N.txt
            if (step == time_steps_to_save[0] || step == time_steps_to_save[1] || step == time_steps_to_save[2] || step == time_steps_to_save[3] || step == time_steps_to_save[4]) {
                std::ofstream outFile;
                if (step == time_steps_to_save[0])
                    outFile.open("results_" + std::to_string(N) + ".txt");
                else 
                    outFile.open("results_" + std::to_string(N) + ".txt", std::ios_base::app);
                for (int i = 0; i < N; ++i) {
                    outFile << u(i + Nbuf) << " ";
                }
                outFile << "\n";
                outFile.close();
            }

            // Calculate error
            double error = 0.0;
            Kokkos::parallel_reduce("Error", N, KOKKOS_LAMBDA(const int i, double& error) {
                double x = i * dx;
                int i_new = i + Nbuf;
                double diff = std::abs(u(i_new) - std::exp(-4.0 * M_PI * M_PI * alpha * step * dt) * std::sin(2.0 * M_PI * x));
                error += diff;
            }, error);
            errors(step) = error/N;
            error_total += error/N;

            double delta[4] = {0, 1, 1.0/3, 1};
            double a[3] = {0, -4.0, 1.0/27};
            double b[3] = {1, 2.0/9, 3.0/4};

            Kokkos::View<double*> q("q", N);

            // Kokkos::fence();

            for (int m = 0; m < 3; m++) {

                Kokkos::View<double*> u_temp("u_temp", N + 2*Nbuf);

                Kokkos::parallel_for("Diffusion", N, KOKKOS_LAMBDA(const int i) {
                    int i_new = i + Nbuf;
                    q(i) = a[m]*q(i) + alpha * (2*u_frac(i_new-3) - 27*u_frac(i_new-2) + 270*u_frac(i_new-1) - 490*u_frac(i_new) + 270*u_frac(i_new+1) - 27*u_frac(i_new+2) + 2*u_frac(i_new+3)) / (180*dx*dx) * dt;
                    u_temp(i_new) = u_frac(i_new) + b[m]*q(i);

                });

                Kokkos::fence();

                // copy u_temp to u_frac
                Kokkos::parallel_for("Copy", N+2*Nbuf, KOKKOS_LAMBDA(const int i) {
                    u_frac(i) = u_temp(i);
                });

                // Update ghost cells
                update_ghost_cells(u_frac, N);

                Kokkos::fence();

            }

            // copy u_frac to u
            Kokkos::parallel_for("Copy", N+2*Nbuf, KOKKOS_LAMBDA(const int i) {
                u(i) = u_frac(i);
            });

            Kokkos::fence();

            // // Note: You may want to synchronize Kokkos view data if needed
            // Kokkos::fence();
        }


        // Write errors to a file
        std::ofstream outFile2("error_with_time.txt");
        for (int step = 0; step < num_steps; ++step) {
            outFile2 << errors(step) << "\n";
        }
        outFile2.close();

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
