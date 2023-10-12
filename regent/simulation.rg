import "regent"

local format = require("std/format")
local m = terralib.includec("math.h")
local c = terralib.includec("stdio.h")

local N = 128            -- Number of spatial grid points
local L = 1.0            -- Length of the rod
local T = 1.0            -- Total time
local alpha = 1          -- Thermal diffusivity

local dx = L / N
local dt = 0.01 
local Nt = m.ceil(T / dt)

local num_partitions = 8
local partition_size = N / num_partitions

-- Define the task for updating the temperature field
task update_field(
                  left_region: region(ispace(int1d, partition_size), double),
                  u: region(ispace(int1d, partition_size), double),
                  right_region: region(ispace(int1d, partition_size), double),
                  u_new: region(ispace(int1d, partition_size), double))
where writes(u_new), reads(left_region, u, right_region) do
    u_new[0] = u[0] + alpha * dt / (dx * dx) * (left_region[partition_size-1] - 2 * u[0] + u[1])
    for i = 1, partition_size-1 do
        u_new[i] = u[i] + alpha * dt / (dx * dx) * (u[i-1] - 2 * u[i] + u[i+1])
    end
    u_new[partition_size-1] = u[partition_size-1] + alpha * dt / (dx * dx) * (u[partition_size-2] - 2 * u[partition_size-1] + right_region[0])
end

task init(u: region(ispace(int1d, N), double))
where writes(u) do
    for i = 0, N do
        u[i] = m.sin(2.0 * m.M_PI * i * dx) -- Set the initial condition here
    end
end

task overwrite(u: region(ispace(int1d, N), double), 
               u_new: region(ispace(int1d, N), double))
where writes(u), reads(u_new) do
    for i = 0, partition_size do
        u[i] = u_new[i]
    end
end

task main()
    -- Create a partitioned region for the temperature field
    -- and initialize the initial condition

    var is_x = ispace(int1d, N)
    var ps = ispace(int1d, num_partitions)

    var u = region(is_x, double)
    var u_p = partition(equal, u, ps)

    var u_new = region(is_x, double)
    var u_new_p = partition(equal, u_new, ps)

    init(u)

    format.println("Initialized the temperature field")

    for i in ps do
        for j = 0, partition_size do
            format.println("u = {}", u_p[i][j])
        end
    end

    -- Time-stepping loop
    for n = 0, Nt do
        -- Update the temperature field in parallel
        for i = 0, num_partitions do
            var left = (i - 1 + num_partitions) % num_partitions
            var right = (i + 1) % num_partitions
            update_field(u_p[left], u_p[i], u_p[right], u_new_p[i])
        end

        for i in ps do
            overwrite(u_p[i], u_new_p[i])
        end
        format.println("Completed iteration {}", n)
    end 

    -- Print the final temperature field
    for i in ps do
        for j = 0, partition_size do
            format.println("u = {}", u_p[i][j])
        end
    end

end

regentlib.start(main)