import "regent"

local format = require("std/format")
local m = terralib.includec("math.h")

local N = 128            -- Number of spatial grid points
local L = 1.0            -- Length of the rod
local T = 1.0            -- Total time
local alpha = 1          -- Thermal diffusivity

local dx = L / N
local dt = 0.00001 
local Nt = m.ceil(T / dt)

local is_x = ispace(int1d, N)

-- Define the task for updating the temperature field
task update_field(u: region(is_x, double), u_new: region(is_x, double))
where writes(u_new), reads(u) do
    var is_x = ispace(int1d, N)
    for i in is_x do
        var left = (i - 1 + N) % N
        var right = (i + 1) % N
        u_new[i] = u[i] + alpha * dt / (dx * dx) * (u[left] - 2.0 * u[i] + u[right])
    end
end

task main()
    -- Create a partitioned region for the temperature field
    -- and initialize the initial condition

    var is_x = ispace(int1d, N)
    var u = region(is_x, double)
    var u_new = region(is_x, double)
    for i in is_x do
        var i_double: double = i
        u[i] = m.sin(2.0 * m.M_PI * i_double * dx) -- Set the initial condition here
        u_new[i] = 0.0
    end

    -- Time-stepping loop
    for n = 1, Nt do
        -- Update the temperature field in parallel
        update_field(u, u_new)
        for i in is_x do 
            u[i] = u_new[i]
        end

        if n == 1 or n == 100 or n == 500 or n == 1000 or n == 5000 or n == 7500 or n == 10000 then
            var error = 0.0
            for i in is_x do
                var i_double: double = i
                error = error + 100000000 * m.fabs(u[i] - m.exp(-4.0 * m.M_PI * m.M_PI * alpha * n * dt) * m.sin(2.0 * m.M_PI * i_double * dx))
            end
            error = error / N
            format.println("Error after {} steps is {}*10^-9", n, error)
        end
    end
    format.println("Done!")

end

regentlib.start(main)