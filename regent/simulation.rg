import "regent"

local m = terralib.includec("math.h")
local c = terralib.includec("stdio.h")

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
        u_new[i] = u[i] + alpha * (u[i+1] - 2.0 * u[i] + u[i-1])
    end
end

task main()
    -- Create a partitioned region for the temperature field
    -- and initialize the initial condition
    var is_x = ispace(int1d, N)
    var u = region(is_x, double)
    var u_new = region(is_x, double)
    for i in is_x do
        u[i] = m.sin(2.0 * m.M_PI * i * dx) -- Set the initial condition here
        u_new[i] = 0.0
    end

    -- Time-stepping loop
    for n = 1, Nt+1 do
        -- Update the temperature field in parallel
        update_field(u, u_new)
    end

    -- Print the final temperature field (u)
    for i in is_x do
        c.printf("Temperature at position %d: %d\n", i * dx, u[i])
    end
end

regentlib.start(main)