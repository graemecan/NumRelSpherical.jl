function get_rhs(state, p, t)

    # ODEProblem requires inputs to be u, p, t returning du
    # The tuple p contains all the extra stuff we need
    r = p[1]
    dr = p[2] 
    drn_matrix = p[3]
    advec_x_matrix = p[4]
    eta = p[5]

    rhs = zeros(size(state))

    u       = view(state, idx_u,       :)
    v       = view(state, idx_v,       :)
    phi     = view(state, idx_phi,     :)
    hrr     = view(state, idx_hrr,     :)
    htt     = view(state, idx_htt,     :)
    hpp     = view(state, idx_hpp,     :)
    K       = view(state, idx_K,       :)
    arr     = view(state, idx_arr,     :)
    att     = view(state, idx_att,     :)
    app     = view(state, idx_app,     :)
    lambdar = view(state, idx_lambdar, :)
    shiftr  = view(state, idx_shiftr,  :)
    br      = view(state, idx_br,      :)
    lapse   = view(state, idx_lapse,   :)

    rhs_u       = view(rhs, idx_u,       :)
    rhs_v       = view(rhs, idx_v,       :)
    rhs_phi     = view(rhs, idx_phi,     :)
    rhs_hrr     = view(rhs, idx_hrr,     :)
    rhs_htt     = view(rhs, idx_htt,     :)
    rhs_hpp     = view(rhs, idx_hpp,     :)
    rhs_K       = view(rhs, idx_K,       :)
    rhs_arr     = view(rhs, idx_arr,     :)
    rhs_att     = view(rhs, idx_att,     :)
    rhs_app     = view(rhs, idx_app,     :)
    rhs_lambdar = view(rhs, idx_lambdar, :)
    rhs_shiftr  = view(rhs, idx_shiftr,  :)
    rhs_br      = view(rhs, idx_br,      :)
    rhs_lapse   = view(rhs, idx_lapse,   :)

    h = vcat(transpose(hrr), transpose(htt), transpose(hpp))
    determinant = abs.(get_rescaled_determinant_gamma(h))

    hrr = (1.0 .+ hrr) ./ determinant.^(1.0/3.) .- 1.0
    htt = (1.0 .+ htt) ./ determinant.^(1.0/3.) .- 1.0
    hpp = (1.0 .+ hpp) ./ determinant.^(1.0/3.) .- 1.0

    second_derivative_indices = [idx_u, idx_phi, idx_hrr, idx_htt,
                                 idx_hpp, idx_shiftr, idx_lapse]
    d2state_dr2 = get_second_derivative(state, second_derivative_indices,
                                        drn_matrix, dr)

    d2u_dr2      = view(d2state_dr2, idx_u,      :)
    d2phi_dr2    = view(d2state_dr2, idx_phi,    :)
    d2hrr_dr2    = view(d2state_dr2, idx_hrr,    :)
    d2htt_dr2    = view(d2state_dr2, idx_htt,    :)
    d2hpp_dr2    = view(d2state_dr2, idx_hpp,    :)
    d2shiftr_dr2 = view(d2state_dr2, idx_shiftr, :)
    d2lapse_dr2  = view(d2state_dr2, idx_lapse,  :)

    first_derivative_indices = [idx_u, idx_phi, idx_hrr, idx_htt,
                                idx_hpp, idx_K, idx_lambdar, 
                                idx_shiftr, idx_lapse]
    dstate_dr = get_first_derivative(state, first_derivative_indices,
                                     drn_matrix, dr)

    du_dr       = view(dstate_dr, idx_u,       :)
    dphi_dr     = view(dstate_dr, idx_phi,     :)
    dhrr_dr     = view(dstate_dr, idx_hrr,     :)
    dhtt_dr     = view(dstate_dr, idx_htt,     :)
    dhpp_dr     = view(dstate_dr, idx_hpp,     :)
    dK_dr       = view(dstate_dr, idx_K,       :)
    dlambdar_dr = view(dstate_dr, idx_lambdar, :)
    dshiftr_dr  = view(dstate_dr, idx_shiftr,  :)
    dlapse_dr   = view(dstate_dr, idx_lapse,   :)

    advec_indices = [idx_u, idx_v, idx_phi, idx_hrr, idx_htt,
                     idx_hpp, idx_arr, idx_att, idx_app, idx_K, 
                     idx_lambdar]
    dstate_dr_advec = get_advection(state, (shiftr .>= 0).+1,
                                    advec_indices, advec_x_matrix, dr)

    du_dr_advec       = view(dstate_dr_advec, idx_u,       :)
    dv_dr_advec       = view(dstate_dr_advec, idx_v,       :)
    dphi_dr_advec     = view(dstate_dr_advec, idx_phi,     :)
    dhrr_dr_advec     = view(dstate_dr_advec, idx_hrr,     :)
    dhtt_dr_advec     = view(dstate_dr_advec, idx_htt,     :)
    dhpp_dr_advec     = view(dstate_dr_advec, idx_hpp,     :)
    darr_dr_advec     = view(dstate_dr_advec, idx_arr,     :)
    datt_dr_advec     = view(dstate_dr_advec, idx_att,     :)
    dapp_dr_advec     = view(dstate_dr_advec, idx_app,     :)
    dK_dr_advec       = view(dstate_dr_advec, idx_K,       :)
    dlambdar_dr_advec = view(dstate_dr_advec, idx_lambdar, :)

    a = vcat(transpose(arr), transpose(att), transpose(app))
    em4phi = exp.(-4.0 .* phi)
    dhdr   = vcat(transpose(dhrr_dr), transpose(dhtt_dr), transpose(dhpp_dr))
    d2hdr2 = vcat(transpose(d2hrr_dr2), transpose(d2htt_dr2), transpose(d2hpp_dr2))

    # Calculate tensor quantities
    r_gamma_LL = get_rescaled_metric(h)
    r_gamma_UU = get_rescaled_inverse_metric(h)

    a_UU     = get_a_UU(a, r_gamma_UU)
    traceA   = get_trace_A(a, r_gamma_UU)
    Asquared = get_Asquared(a, r_gamma_UU)

    rDelta_U, rDelta_ULL, rDelta_LLL = get_rescaled_connection(r, r_gamma_UU,
                                                               r_gamma_LL, h, dhdr)

    r_conformal_chris = get_rescaled_conformal_chris(rDelta_ULL, r)
    rbar_Rij = get_rescaled_ricci_tensor(r, h, dhdr, d2hdr2, lambdar, dlambdar_dr,
                                         rDelta_U, rDelta_ULL, rDelta_LLL, 
                                         r_gamma_UU, r_gamma_LL)

    # Conformal divergence of the shift
    bar_div_shift = (dshiftr_dr + 2.0 .* shiftr ./ r)

    # Matter sources
    matter_rho = get_rho(u, du_dr, v, r_gamma_UU, em4phi)
    matter_Si  = get_Si(u, du_dr, v)
    matter_S, matter_rSij = get_rescaled_Sij(u, du_dr, v, r_gamma_UU, 
                                             em4phi, r_gamma_LL)

    #--------- RHS Calculation in earnest!--------------------------
    rhs_u[:], rhs_v[:] = get_matter_rhs(u, v, du_dr, d2u_dr2,
                                        r_gamma_UU, em4phi, dphi_dr,
                                        K, lapse, dlapse_dr, r_conformal_chris, dhdr)

    rhs_phi[:] = get_rhs_phi(lapse, K, bar_div_shift)

    rhs_h      = get_rhs_h(r, r_gamma_LL, lapse, traceA, dshiftr_dr,
                           shiftr, bar_div_shift, a)

    rhs_K[:]   = get_rhs_K(lapse, K, Asquared, em4phi, d2lapse_dr2,
                         dlapse_dr, r_conformal_chris, dphi_dr, r_gamma_UU,
                         matter_rho, matter_S)

    rhs_a      = get_rhs_a(r, a, bar_div_shift, lapse, K, em4phi, rbar_Rij,
                           r_conformal_chris, r_gamma_UU, r_gamma_LL,
                           d2phi_dr2, dphi_dr, d2lapse_dr2, dlapse_dr,
                           matter_rSij)

    rhs_lambdar[:] = get_rhs_lambdar(r, d2shiftr_dr2, dshiftr_dr, shiftr,
                                     rDelta_U, rDelta_ULL, bar_div_shift, 
                                     r_gamma_UU, a_UU, lapse, dlapse_dr, 
                                     dphi_dr, dK_dr, matter_Si)

    # Gauge rhs
    rhs_br[:]     = 0.75 .* rhs_lambdar - eta .* br # Eqn 14b Etienne
    rhs_shiftr[:] = br # Eqn 14a Etienne / Eqn 16a Baumgarte
    rhs_lapse[:]  = -2.0 .* lapse .* K # 1+log slicing

    # Advection terms coming from Lie derivative along shift
    # (Eqn 8 Baumgarte)
    rhs_u[:] .+= shiftr .* du_dr_advec
    rhs_v[:] .+= shiftr .* dv_dr_advec
    rhs_phi[:] .+= shiftr .* dphi_dr_advec
    rhs_hrr[:] = rhs_h[i_r, i_r, :] + shiftr .* dhrr_dr_advec + 2.0 .* hrr .* dshiftr_dr
    rhs_htt[:] = rhs_h[i_t, i_t, :] + shiftr .* dhtt_dr_advec + 2.0 .* shiftr .* 1.0 ./ r .* htt
    rhs_hpp[:] = rhs_h[i_p, i_p, :] + shiftr .* dhpp_dr_advec + 2.0 .* shiftr .* 1.0 ./ r .* hpp
    rhs_K[:] .+= shiftr .* dK_dr_advec
    rhs_arr[:] = rhs_a[i_r, i_r, :] + shiftr .* darr_dr_advec + 2.0 .* arr .* dshiftr_dr
    rhs_att[:] = rhs_a[i_t, i_t, :] + shiftr .* datt_dr_advec + 2.0 .* shiftr .* 1.0 ./ r .* att
    rhs_app[:] = rhs_a[i_p, i_p, :] + shiftr .* dapp_dr_advec + 2.0 .* shiftr .* 1.0 ./ r .* app
    rhs_lambdar[:] .+= shiftr .* dlambdar_dr_advec - lambdar .* dshiftr_dr

    # Kreiss-Oliger (not currently implemented in Engrenage)
    sigma = 0.0

    diss_indices = [idx_u, idx_v, idx_phi, idx_hrr, idx_htt,
                    idx_hpp, idx_K, idx_arr, idx_att, idx_app,
                    idx_lambdar, idx_shiftr, idx_br, idx_lapse]

    diss = sigma .* get_kreiss_oliger_diss(state, 
                                           diss_indices, 
                                           drn_matrix, dr)

    rhs .+= sigma .* diss

    fill_outer_boundary!(rhs, ALL_INDICES, r)
    fill_inner_boundary!(rhs, ALL_INDICES)

    return rhs

end