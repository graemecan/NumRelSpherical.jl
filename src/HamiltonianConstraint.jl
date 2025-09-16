import DifferentialEquations as DE

A = 1.0
σ = 1.0
m = 1.0

tspan = (0.0, 1.0)

function hamiltonianConstraint!(du, u, p, t)
    ϕ = u[1]
    dϕ = u[2]
    du[1] = dϕ
    du[2] = (2/t)*dϕ - dϕ^2 - π*((4*t^2)/σ^2)*A^2*exp(-(2*t^2)/σ^2)
end

function bc!(residual, u, p, t)
    residual[1] = u(1)[1]
    residual[2] = u(0)[2]
end

bvp = DE.BVProblem(hamiltonianConstraint!, bc!, [1.0, 1.0], tspan)
sol = DE.solve(bvp, DE.MIRK4(); dt = 0.05)