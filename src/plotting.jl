using Plots

# Run this after the "run_model.jl" script
# Assumes the ODE has been solved and saved in the "sol" structure
function plot_field(field_idx, t, lower_limit, upper_limit)

    plot(r, sol.u[t][field_idx, :], ylimits=(lower_limit, upper_limit))

end

total_t = length(sol.u)

Δt = 5
field_idx = nr.idx_u
limits = 1e-4

@gif for t ∈ 1:total_t
    plot_field(field_idx, t, -0.1, 0.1)
end every Δt