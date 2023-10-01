import "regent"

local N = 128            -- Number of spatial grid points
local L = 1.0            -- Length of the rod
local T = 1.0            -- Total time
local alpha = 1          -- Thermal diffusivity

local dx = L / N
local dt = 0.00001 
local Nt = T / dt

-- Define the task for updating the temperature field
task update_field(u: region(ispace, double), u_new: region(ispace, double))
  --[write(u), write(u_new)]
    for i = 0, N do
    u[i] = u_new[i] + alpha * (u_new[i+1] - 2.0 * u_new[i] + u_new[i-1])
    end
end

task main()
    -- Create a partitioned region for the temperature field
    -- and initialize the initial condition
    var u: region(ispace, double)
    var u_new: region(ispace, double)
    for i = 0, N do
        u[i] = sin(2.0 * pi() * i * dx) -- Set the initial condition here
        u_new[i] = 0.0
    end

    -- Time-stepping loop
    for n = 1, Nt+1 do
        -- Swap the regions for each time step
        var tmp: region(ispace, double)
        tmp = u
        u = u_new
        u_new = tmp
        -- Update the temperature field in parallel
        update_field(u, u_new)
    end

    -- Print the final temperature field (u)
    for i = 0, N do
        writeln("Temperature at position ", i * dx, ": ", u[i])
    end
end

regentlib.start(main)