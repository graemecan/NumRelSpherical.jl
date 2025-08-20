import DifferentialEquations as DE

const g = 9.81
L = 1.0

tspan = (0.0, pi/2)

function simplependulum!(du, u, p, t)
    θ = u[1]
    dθ = u[2]
    du[1] = dθ
    du[2] = -(g/L)*sin(θ)
end

function bc!(residual, u, p, t)
    residual[1] = u(pi/4)[1] + pi/2
    residual[2] = u(pi/2)[1] - pi/2
end

bvp = DE.BVProblem(simplependulum!, bc!, [pi/2, pi/2], tspan)
sol = DE.solve(bvp, DE.MIRK4(); dt = 0.05)