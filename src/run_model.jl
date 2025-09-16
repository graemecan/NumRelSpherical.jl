# To run this directly in the Julia REPL (without VS Code) do the following:
# ] activate NumRelSpherical (in the source directory?)
# ] dev path/to/package
# > import NumRelSpherical as nr
# Then it precompiles a bunch of shit

import NumRelSpherical as nr
using ProgressLogging

params = nr.get_spacing_parameters(96.0,1/16,2,"Cubic");

r, dr, dx, dnr_dxn = nr.get_spacing(params);

dxn_matrix = nr.get_dxn_matrix(params[:num_points]);
drn_matrix = nr.get_drn_matrix(dx, dnr_dxn, params[:num_points], dxn_matrix);
advec_x_matrix = nr.compute_advec_x_matrix(params[:num_points]);

initial_state = nr.get_scalar_collapse_initial_state(r, params[:num_points], drn_matrix, dr,
                                                     3.7e-2,3.0);

eta = 2.0;
p = (r, dr, drn_matrix, advec_x_matrix, eta);

using DifferentialEquations

tspan = (0.0, 48.0);
prob = ODEProblem(nr.get_rhs, initial_state, tspan, p);
#sol = solve(prob, progress=true, progress_steps=10);
