import .NumRelSpherical as nr
using ProgressLogging

params = nr.get_spacing_parameters(96.0,1/16,2,"Cubic");

r, dr, dx, dnr_dxn = nr.get_spacing(params);

dxn_matrix = nr.get_dxn_matrix(params[:num_points]);
drn_matrix = nr.get_drn_matrix(dx, dnr_dxn, params[:num_points], dxn_matrix);
advec_x_matrix = nr.compute_advec_x_matrix(params[:num_points]);

initial_state = nr.get_scalar_collapse_initial_state(r, params[:num_points], drn_matrix, dr,
                                                     3.7e-3,20.0,3.0);

eta = 2.0;

u       = view(initial_state, nr.idx_u,       :)
v       = view(initial_state, nr.idx_v,       :)
phi     = view(initial_state, nr.idx_phi,     :)
hrr     = view(initial_state, nr.idx_hrr,     :)
htt     = view(initial_state, nr.idx_htt,     :)
hpp     = view(initial_state, nr.idx_hpp,     :)
K       = view(initial_state, nr.idx_K,       :)
arr     = view(initial_state, nr.idx_arr,     :)
att     = view(initial_state, nr.idx_att,     :)
app     = view(initial_state, nr.idx_app,     :)
lambdar = view(initial_state, nr.idx_lambdar, :)
shiftr  = view(initial_state, nr.idx_shiftr,  :)
br      = view(initial_state, nr.idx_br,      :)
lapse   = view(initial_state, nr.idx_lapse,   :)

h = vcat(transpose(hrr), transpose(htt), transpose(hpp));
determinant = abs.(nr.get_rescaled_determinant_gamma(h));

hrr = (1.0 .+ hrr) ./ determinant.^(1.0/3.) .- 1.0;
htt = (1.0 .+ htt) ./ determinant.^(1.0/3.) .- 1.0;
hpp = (1.0 .+ hpp) ./ determinant.^(1.0/3.) .- 1.0;

second_derivative_indices = [nr.idx_u, nr.idx_phi, nr.idx_hrr, nr.idx_htt, nr.idx_hpp, nr.idx_shiftr, nr.idx_lapse];

d2state_dr2 = nr.get_second_derivative(initial_state, second_derivative_indices, drn_matrix, dr);

d2u_dr2      = view(d2state_dr2, nr.idx_u,      :)
d2phi_dr2    = view(d2state_dr2, nr.idx_phi,    :)
d2hrr_dr2    = view(d2state_dr2, nr.idx_hrr,    :)
d2htt_dr2    = view(d2state_dr2, nr.idx_htt,    :)
d2hpp_dr2    = view(d2state_dr2, nr.idx_hpp,    :)
d2shiftr_dr2 = view(d2state_dr2, nr.idx_shiftr, :)
d2lapse_dr2  = view(d2state_dr2, nr.idx_lapse,  :)

first_derivative_indices = [nr.idx_u, nr.idx_phi, nr.idx_hrr, nr.idx_htt, nr.idx_hpp, nr.idx_K, nr.idx_lambdar,  nr.idx_shiftr, nr.idx_lapse];

dstate_dr = nr.get_first_derivative(initial_state, first_derivative_indices, drn_matrix, dr);

du_dr       = view(dstate_dr, nr.idx_u,       :)
dphi_dr     = view(dstate_dr, nr.idx_phi,     :)
dhrr_dr     = view(dstate_dr, nr.idx_hrr,     :)
dhtt_dr     = view(dstate_dr, nr.idx_htt,     :)
dhpp_dr     = view(dstate_dr, nr.idx_hpp,     :)
dK_dr       = view(dstate_dr, nr.idx_K,       :)
dlambdar_dr = view(dstate_dr, nr.idx_lambdar, :)
dshiftr_dr  = view(dstate_dr, nr.idx_shiftr,  :)
dlapse_dr   = view(dstate_dr, nr.idx_lapse,   :)

advec_indices = [nr.idx_u, nr.idx_v, nr.idx_phi, nr.idx_hrr, nr.idx_htt, nr.idx_hpp, nr.idx_arr, nr.idx_att, nr.idx_app, nr.idx_K, nr.idx_lambdar];

dstate_dr_advec = nr.get_advection(initial_state, (shiftr .>= 0).+1, advec_indices, advec_x_matrix, dr);

du_dr_advec       = view(dstate_dr_advec, nr.idx_u,       :)
dv_dr_advec       = view(dstate_dr_advec, nr.idx_v,       :)
dphi_dr_advec     = view(dstate_dr_advec, nr.idx_phi,     :)
dhrr_dr_advec     = view(dstate_dr_advec, nr.idx_hrr,     :)
dhtt_dr_advec     = view(dstate_dr_advec, nr.idx_htt,     :)
dhpp_dr_advec     = view(dstate_dr_advec, nr.idx_hpp,     :)
darr_dr_advec     = view(dstate_dr_advec, nr.idx_arr,     :)
datt_dr_advec     = view(dstate_dr_advec, nr.idx_att,     :)
dapp_dr_advec     = view(dstate_dr_advec, nr.idx_app,     :)
dK_dr_advec       = view(dstate_dr_advec, nr.idx_K,       :)
dlambdar_dr_advec = view(dstate_dr_advec, nr.idx_lambdar, :)

a = vcat(transpose(arr), transpose(att), transpose(app));
em4phi = exp.(-4.0 .* phi);
dhdr   = vcat(transpose(dhrr_dr), transpose(dhtt_dr), transpose(dhpp_dr));
d2hdr2 = vcat(transpose(d2hrr_dr2), transpose(d2htt_dr2), transpose(d2hpp_dr2));

# Calculate tensor quantities
r_gamma_LL = nr.get_rescaled_metric(h);
r_gamma_UU = nr.get_rescaled_inverse_metric(h);

a_UU     = nr.get_a_UU(a, r_gamma_UU);
traceA   = nr.get_trace_A(a, r_gamma_UU);
Asquared = nr.get_Asquared(a, r_gamma_UU);

rDelta_U, rDelta_ULL, rDelta_LLL = nr.get_rescaled_connection(r, r_gamma_UU, r_gamma_LL, h, dhdr);

r_conformal_chris = nr.get_rescaled_conformal_chris(rDelta_ULL, r);