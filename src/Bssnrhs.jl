# Eqn 9c (1211.6632) (typo in paper, should be same index on Dbar as on beta, it's a divergence)
function get_rhs_phi(lapse, K, bar_div_shift)

    dphidt = -one_sixth .* lapse .* K + one_sixth .* bar_div_shift

    return dphidt

end

# Eqn 11a from Ruchlin, Etienne, Baumgarte (1712.07658)
# This differs from the analogous equation 9a of Baumgarte
# by a term proportional to the trace of A. This somehow
# dynamically enforces the trace-free condition on A. There's also
# the symmetrised spatial covariant derivative of beta in this
# version of the equation (rhat_D_shift)
# Note that rescaled quantities are used throughout!
function get_rhs_h(r, r_gamma_LL, lapse, traceA, dshiftrdx, shiftr, bar_div_shift, a)

    N = size(r)[1]
    
    rflat_chris = get_rescaled_flat_spherical_chris(r)
    rhat_D_shift = zeros((SPACEDIM, SPACEDIM, N))
    rhat_D_shift[i_r, i_r, :] = dshiftrdx
    rhat_D_shift[i_t, i_t, :] = rflat_chris[i_t, i_t, i_r, :] .* shiftr
    rhat_D_shift[i_p, i_p, :] = rflat_chris[i_p, i_p, i_r, :] .* shiftr

    dhdt = zeros((SPACEDIM, SPACEDIM, N))

    for i in 1:SPACEDIM
        dhdt[i, i, :] .+= (two_thirds .* r_gamma_LL[i, :]
                          .* (lapse .* traceA .- bar_div_shift)
                          .- 2.0 .* lapse .* a[i, :])

        for j in 1:SPACEDIM
            dhdt[i, j, :] .+= rhat_D_shift[i, j, :] .+ rhat_D_shift[j, i, :]
        end
    end

    return dhdt

end

# Eqn 9d Baumgarte (11d Etienne + matter terms)
function get_rhs_K(lapse, K, Asquared, em4phi, d2lapsedr2, dlapsedr, r_conformal_chris, dphidr, r_gamma_UU, rho, S)

    bar_D2_lapse = (r_gamma_UU[i_r, :] .* (d2lapsedr2[:] .- r_conformal_chris[i_r, i_r, i_r, :] .* dlapsedr)
                  .- r_gamma_UU[i_t, :] .* r_conformal_chris[i_r, i_t, i_t, :] .* dlapsedr[:]
                  .- r_gamma_UU[i_p, :] .* r_conformal_chris[i_r, i_p, i_p, :] .* dlapsedr[:])

    dKdt = (one_third .* lapse .* K .* K
          .+ lapse .* Asquared
          .- em4phi .* (bar_D2_lapse .+ 2.0 .* r_gamma_UU[i_r, :] .* dlapsedr .* dphidr)
          .+ 0.5 .* eight_pi_G .* lapse .* (rho + S))

    return dKdt

end

# Eqn 9b. Note that the quantity in ^TF of this equation is just
# fully calculated, and then the trace of that is calculated and 
# multiplied by 1/3*gamma_{ij} and then subtracted, to construct the
# trace-free part.
function get_rhs_a(r, a, bar_div_shift, lapse, K, em4phi, rbar_Rij, r_conformal_chris, r_gamma_UU, r_gamma_LL, d2phidr2, dphidr, d2lapsedr2, dlapsedr, rSij)

    N = size(r)[1]

    r_dAdt_TF_part = zeros((SPACEDIM, N))
    r_AikAkj       = zeros((SPACEDIM, N))

    r_AikAkj[i_r, :] = a[i_r, :] .* a[i_r, :] .* r_gamma_UU[i_r, :]
    r_AikAkj[i_t, :] = a[i_t, :] .* a[i_t, :] .* r_gamma_UU[i_t, :]
    r_AikAkj[i_p, :] = a[i_p, :] .* a[i_p, :] .* r_gamma_UU[i_p, :]

    r_dAdt_TF_part[i_r, :] = (-2.0 .* lapse .* d2phidr2 + 4.0 .* lapse .* dphidr .* dphidr
                                                        + 4.0 .* dlapsedr .* dphidr
                                                        - d2lapsedr2)

    r_dAdt_TF_part[i_r, :] += (2.0 .* lapse .* dphidr + dlapsedr) .* r_conformal_chris[i_r, i_r, i_r, :]
    r_dAdt_TF_part[i_t, :] = (2.0 .* lapse .* dphidr + dlapsedr) .* r_conformal_chris[i_r, i_t, i_t, :]
    r_dAdt_TF_part[i_p, :] = (2.0 .* lapse .* dphidr + dlapsedr) .* r_conformal_chris[i_r, i_p, i_p, :]

    for i in 1:SPACEDIM
        r_dAdt_TF_part[i, :] += lapse .* (rbar_Rij[i, i, :] - eight_pi_G .* rSij[i, i, :])
    end

    r_fullgamma_UU = zeros((SPACEDIM, N))
    r_fullgamma_UU[1, :] = em4phi .* r_gamma_UU[1, :]
    r_fullgamma_UU[2, :] = em4phi .* r_gamma_UU[2, :]
    r_fullgamma_UU[3, :] = em4phi .* r_gamma_UU[3, :]

    trace = get_trace(r_dAdt_TF_part, r_fullgamma_UU)

    dadt = zeros((SPACEDIM, SPACEDIM, N))
    for i in 1:SPACEDIM
        dadt[i, i, :] += (- two_thirds .* a[i, :] .* bar_div_shift
                          - 2.0 .* lapse .* r_AikAkj[i, :]
                          + lapse .* a[i, :] .* K
                          + em4phi .* r_dAdt_TF_part[i, :]
                          - one_third .* trace .* r_gamma_LL[i, :])
    end
    
    return dadt

end

# Eqn 9e Baumgarte
function get_rhs_lambdar(r, d2shiftrdr2, dshiftrdr, shiftr, rDelta_U, rDelta_ULL, bar_div_shift, r_gamma_UU, a_UU, lapse, dlapsedr, dphidr, dKdr, Si)

    rflat_chris = get_rescaled_flat_spherical_chris(r)

    hat_D2_shiftr = (r_gamma_UU[i_r, :] .* d2shiftrdr2
                   - r_gamma_UU[i_t, :] .* rflat_chris[i_r, i_t, i_t, :] .* dshiftrdr
                   - r_gamma_UU[i_p, :] .* rflat_chris[i_r, i_p, i_p, :] .* dshiftrdr
                   + (r_gamma_UU[i_t, :] .* rflat_chris[i_r, i_t, i_t, :]
                                            .* rflat_chris[i_t, i_r, i_t, :] .* shiftr)
                   + (r_gamma_UU[i_p, :] .* rflat_chris[i_r, i_p, i_p, :]
                                            .* rflat_chris[i_p, i_r, i_p, :] .* shiftr))
    
    bar_D_div_shift = r_gamma_UU[i_r, :] .* (d2shiftrdr2 + 2.0 ./ r .* dshiftrdr - 2.0 ./ r ./ r .* shiftr)

    dlambdardt = (hat_D2_shiftr
                  + two_thirds .* rDelta_U[i_r, :] .* bar_div_shift
                  + one_third .* bar_D_div_shift
                  - 2.0 .* a_UU[i_r, :] .* (dlapsedr - 6.0 .* lapse .* dphidr
                                                     - lapse .* rDelta_ULL[i_r, i_r, i_r, :])
                  + 2.0 .* a_UU[i_t, :] .* lapse .* rDelta_ULL[i_r, i_t, i_t, :]
                  + 2.0 .* a_UU[i_p, :] .* lapse .* rDelta_ULL[i_r, i_p, i_p, :]
                  - four_thirds .* lapse .* r_gamma_UU[i_r, :] .* dKdr
                  - 2.0 .* eight_pi_G .* lapse .* r_gamma_UU[i_r, :] .* Si[i_r, :])

    return dlambdardt

end