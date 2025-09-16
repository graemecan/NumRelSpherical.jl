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

function get_scalar_collapse_initial_state(r, num_points, drn_matrix, dr, amp, siggy, N_for_ham)

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

    u[:] = amp .* exp.(-(r ./ sigma).^2)

    grr = ones(num_points)
    gtt_over_r2 = grr
    gpp_over_r2sintheta = gtt_over_r2
    phys_gamma_over_r4sin2theta = grr .* gtt_over_r2 .* gpp_over_r2sintheta

    # Sign error in Baumgarte Eqn 2
    #phi[:] = (1.0/12.0) .* log.(phys_gamma_over_r4sin2theta)

    r_other_grid, phi_other_grid = solve_hamilton_constraint(r[end], N_for_ham, 1.0, amp, siggy, 1.5e-11, 30)
    interp_linear_extrap = linear_interpolation(r_other_grid, phi_other_grid, extrapolation_bc=Line());
    phi = interp_linear_extrap(r)

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

function solve_hamilton_constraint(R=1.0, N=200, phiR=0.0, A=1.0, sigma=0.5, tol=1e-10, maxits=50)
    dr = R / N
    r = collect(LinRange(0, R, N+1))

    phi = ones(N+1)

    function S(r_val, phi_val)
        return -pi * phi_val * ((4.0*r_val^2)/sigma^4) * (A^2) * exp(-(2.0*r_val^2)/sigma^2)
    end

    function dS_dphi(r_val, phi_val)
        return -pi * ((4.0*r_val^2)/sigma^4) * (A^2) * exp(-(2.0*r_val^2)/sigma^2)
    end
    
    for it in range(1,maxits)
        F = zeros(N+1)

        lap0 = 6.0 * (phi[2]-phi[1]) / dr^2
        F[1] = lap0 - S(r[1], phi[1])

        for i in range(2, N)
            rp = (r[i] + dr/2)^2
            rm = (r[i] - dr/2)^2
            f_plus = rp * (phi[i+1] - phi[i]) / dr
            f_minus = rm * (phi[i] - phi[i-1]) / dr
            lap_i = (f_plus - f_minus) / (dr * r[i]^2)
            F[i] = lap_i - S(r[i], phi[i])
        end

        F[N+1] = phiR - phi[N+1]

        res_norm = norm(F, Inf)
        print("iter: $it ||F||_inf = $res_norm \n")
        if res_norm < tol
            break
        end

        diag_main = zeros(N+1)
        diag_lower = zeros(N)
        diag_upper = zeros(N)

        diag_main[1] = -6.0 / dr^2 - dS_dphi(r[1], phi[1])
        diag_upper[1] = 6.0 / dr^2

        for i in range(2, N)
            rp = (r[i] + dr/2)^2
            rm = (r[i] - dr/2)^2
            denom = dr * r[i]^2
            a_ip1 = rp / (dr * denom)
            a_im1 = rm / (dr * denom)
            a_i = -(rp + rm) / (dr * denom)

            diag_lower[i-1] = a_im1
            diag_main[i] = a_i - dS_dphi(r[i], phi[i])
            diag_upper[i] = a_ip1
        end

        diag_main[N+1] = 1.0

        J = diagm(-1=>diag_lower, 0=>diag_main, 1=>diag_upper)

        dx = J \ -F
        phi .+= dx
    end

    return r, phi
    
end



