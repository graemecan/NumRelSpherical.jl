function get_BH_initial_state(r, num_points, drn_matrix, dr)

    initial_state = zeros((NUM_VARS, num_points))

    u = view(initial_state, idx_u, :)
    v = view(initial_state, idx_v, :)
    phi = view(initial_state, idx_phi, :)
    hrr = view(initial_state, idx_hrr, :)
    htt = view(initial_state, idx_htt, :)
    hpp = view(initial_state, idx_hpp, :)
    K = view(initial_state, idx_K, :)
    arr = view(initial_state, idx_arr, :)
    att = view(initial_state, idx_att, :)
    app = view(initial_state, idx_app, :)
    lambdar = view(initial_state, idx_lambdar, :)
    shiftr = view(initial_state, idx_shiftr, :)
    br = view(initial_state, idx_br, :)
    lapse = view(initial_state, idx_lapse, :)

    # Set BH length scale
    GM = 1.0

    grr = (1.0 .+ 0.5 .* GM ./ r).^4
    gtt_over_r2 = grr
    gpp_over_r2sintheta = gtt_over_r2
    phys_gamma_over_r4sin2theta = grr .* gtt_over_r2 .* gpp_over_r2sintheta

    # Sign error in Baumgarte Eqn 2
    phi[:] = (1.0/12.0) .* log.(phys_gamma_over_r4sin2theta)

    # Cap phi
    phi[:] .= min.(phi, 10.0)
    em4phi = exp.(-4 .* phi)
    hrr[:] = em4phi .* grr .- 1.0
    htt[:] = em4phi .* gtt_over_r2 .- 1.0
    hpp[:] = em4phi .* gpp_over_r2sintheta .- 1.0

    lapse[:] .= 1.0

    fill_inner_boundary!(initial_state, ALL_INDICES)

    h_tensor = vcat(transpose(hrr), transpose(htt), transpose(hpp))
    dh_dr = get_first_derivative(h_tensor, [1], drn_matrix, dr)

    bar_gamma_LL = get_metric(r, h_tensor)
    bar_gamma_UU = get_inverse_metric(r, h_tensor)

    Delta_U, Delta_ULL, Delta_LLL = get_connection(r, bar_gamma_UU, bar_gamma_LL, h_tensor, dh_dr)
    lambdar[:] = Delta_U[i_r, :]

    fill_outer_boundary!(initial_state, [idx_lambdar], r)
    fill_inner_boundary!(initial_state, [idx_lambdar])

    return initial_state

end

function get_flat_initial_state(r, num_points, drn_matrix, dr)

    initial_state = zeros((NUM_VARS, num_points))

    u = view(initial_state, idx_u, :)
    v = view(initial_state, idx_v, :)
    phi = view(initial_state, idx_phi, :)
    hrr = view(initial_state, idx_hrr, :)
    htt = view(initial_state, idx_htt, :)
    hpp = view(initial_state, idx_hpp, :)
    K = view(initial_state, idx_K, :)
    arr = view(initial_state, idx_arr, :)
    att = view(initial_state, idx_att, :)
    app = view(initial_state, idx_app, :)
    lambdar = view(initial_state, idx_lambdar, :)
    shiftr = view(initial_state, idx_shiftr, :)
    br = view(initial_state, idx_br, :)
    lapse = view(initial_state, idx_lapse, :)

    grr = ones(num_points)
    gtt_over_r2 = grr
    gpp_over_r2sintheta = gtt_over_r2
    phys_gamma_over_r4sin2theta = grr .* gtt_over_r2 .* gpp_over_r2sintheta

    # Sign error in Baumgarte Eqn 2
    phi[:] = (1.0/12.0) .* log.(phys_gamma_over_r4sin2theta)

    # Cap phi
    phi[:] .= min.(phi, 10.0)
    em4phi = exp.(-4 .* phi)
    hrr[:] = em4phi .* grr .- 1.0
    htt[:] = em4phi .* gtt_over_r2 .- 1.0
    hpp[:] = em4phi .* gpp_over_r2sintheta .- 1.0

    lapse[:] .= 1.0

    fill_inner_boundary!(initial_state, ALL_INDICES)

    h_tensor = vcat(transpose(hrr), transpose(htt), transpose(hpp))
    dh_dr = get_first_derivative(h_tensor, [1], drn_matrix, dr)

    bar_gamma_LL = get_metric(r, h_tensor)
    bar_gamma_UU = get_inverse_metric(r, h_tensor)

    Delta_U, Delta_ULL, Delta_LLL = get_connection(r, bar_gamma_UU, bar_gamma_LL, h_tensor, dh_dr)
    lambdar[:] = Delta_U[i_r, :]

    fill_outer_boundary!(initial_state, [idx_lambdar], r)
    fill_inner_boundary!(initial_state, [idx_lambdar])

    return initial_state

end

function get_scalar_collapse_initial_state(r, num_points, drn_matrix, dr, amp, mu, sigma)

    initial_state = zeros((NUM_VARS, num_points))

    u = view(initial_state, idx_u, :)
    v = view(initial_state, idx_v, :)
    phi = view(initial_state, idx_phi, :)
    hrr = view(initial_state, idx_hrr, :)
    htt = view(initial_state, idx_htt, :)
    hpp = view(initial_state, idx_hpp, :)
    K = view(initial_state, idx_K, :)
    arr = view(initial_state, idx_arr, :)
    att = view(initial_state, idx_att, :)
    app = view(initial_state, idx_app, :)
    lambdar = view(initial_state, idx_lambdar, :)
    shiftr = view(initial_state, idx_shiftr, :)
    br = view(initial_state, idx_br, :)
    lapse = view(initial_state, idx_lapse, :)

    u[:] = amp .* r.^3 .* exp.(-((r .- mu) ./ sigma).^2)

    grr = ones(num_points)
    gtt_over_r2 = grr
    gpp_over_r2sintheta = gtt_over_r2
    phys_gamma_over_r4sin2theta = grr .* gtt_over_r2 .* gpp_over_r2sintheta

    # Sign error in Baumgarte Eqn 2
    phi[:] = (1.0/12.0) .* log.(phys_gamma_over_r4sin2theta)

    # Cap phi
    phi[:] .= min.(phi, 10.0)
    em4phi = exp.(-4 .* phi)
    hrr[:] = em4phi .* grr .- 1.0
    htt[:] = em4phi .* gtt_over_r2 .- 1.0
    hpp[:] = em4phi .* gpp_over_r2sintheta .- 1.0

    lapse[:] .= 1.0

    fill_inner_boundary!(initial_state, ALL_INDICES)

    h_tensor = vcat(transpose(hrr), transpose(htt), transpose(hpp))
    dh_dr = get_first_derivative(h_tensor, [1], drn_matrix, dr)

    bar_gamma_LL = get_metric(r, h_tensor)
    bar_gamma_UU = get_inverse_metric(r, h_tensor)

    Delta_U, Delta_ULL, Delta_LLL = get_connection(r, bar_gamma_UU, bar_gamma_LL, h_tensor, dh_dr)
    lambdar[:] = Delta_U[i_r, :]

    # As it currently stands this code gives an incorrect initial condition
    # because it doesn't satisfy the Hamiltonian constraint

    fill_outer_boundary!(initial_state, [idx_lambdar], r)
    fill_inner_boundary!(initial_state, [idx_lambdar])

    return initial_state

end