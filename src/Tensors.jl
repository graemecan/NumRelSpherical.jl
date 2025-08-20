# The Christoffel connections in a non-coord basis are given by
# Eq. 7.142 in Nakahara
function get_rescaled_flat_spherical_chris(r)
    N = size(r)[1]
    spherical_chris = zeros((SPACEDIM, SPACEDIM, SPACEDIM, N))
    one_over_r = 1.0 ./ r

    spherical_chris[i_r, i_t, i_t, :] = -one_over_r
    spherical_chris[i_r, i_p, i_p, :] = -one_over_r

    #spherical_chris[i_t, i_p, i_p, :] = -costheta * one_over_r #Original, erroneous
    spherical_chris[i_t, i_p, i_p, :] = -(costheta / sintheta) * one_over_r
    spherical_chris[i_t, i_r, i_t, :] = one_over_r
    spherical_chris[i_t, i_t, i_r, :] = one_over_r

    spherical_chris[i_p, i_p, i_r, :] = one_over_r
    spherical_chris[i_p, i_r, i_p, :] = one_over_r
    spherical_chris[i_p, i_t, i_p, :] = costheta / sintheta * one_over_r
    spherical_chris[i_p, i_p, i_t, :] = costheta / sintheta * one_over_r

    return spherical_chris
end


function get_rescaled_conformal_chris(rDelta_ULL, r)

    return get_rescaled_flat_spherical_chris(r) + rDelta_ULL

end

# This is the determinant of Eqn 19 Baumgarte divided by r^4 sin^2 theta
# assuming that the metric is diagonal.
# Note that the determinant of \bar{\gamma}_{ij} in the non-coordinate
# basis *still* includes the r^4 sin^2 \theta factor, because that comes from
# the basis co-vectors.
function get_rescaled_determinant_gamma(h_tensor)

    determinant = (1.0 .+ h_tensor[i_r, :]) .* (1.0 .+ h_tensor[i_t, :]) .* (1.0 .+ h_tensor[i_p, :])

    return determinant

end

# This is the metric in Eqn 19 with the quantities referred to as
# "scaling" in the function above removed because they are absorbed
# into the basis covectors
function get_rescaled_metric(h_tensor)

    r_gamma_LL = 1.0 .+ h_tensor

    return r_gamma_LL

end

# Inverse of (diagonal) metric in Eqn 19, with scaling factors
# absorbed into basis vectors
function get_rescaled_inverse_metric(h_metric)

    return 1.0 ./ (1.0 .+ h_metric)

end

# Raises indices on a using rescaled metric (i.e. all in non-coord basis)
function get_a_UU(a, r_gamma_UU)

    a_UU = r_gamma_UU .* a .* r_gamma_UU

    return a_UU

end

# Gets the trace of a in the non-coord basis (although traces are
# basis independent!)
function get_trace_A(a, r_gamma_UU)

    return vec(sum(r_gamma_UU .* a, dims=1))

end

# Calculates a trace of a rank-2 tensor, using a given metric
function get_trace(T_LL, gamma_UU)

    return vec(sum(gamma_UU .* T_LL, dims=1))

end

# Calculates A_{ij} A^{ij} using a and rescaled inverse metric
# Again, this is a scalar and therefore basis independent
function get_Asquared(a, r_gamma_UU)

    return vec(sum(a .* a .* r_gamma_UU .* r_gamma_UU, dims=1))

end

# Calculates connection in the non-coordinate basis
function get_rescaled_connection(r, r_gamma_UU, r_gamma_LL, h, dhdr)

    N = size(r)[1]
    rDelta_ULL = zeros((SPACEDIM, SPACEDIM, SPACEDIM, N))
    rDelta_LLL = zeros((SPACEDIM, SPACEDIM, SPACEDIM, N))
    rhat_D_bar_gamma = get_rescaled_hat_D_bar_gamma(r, h, dhdr)

    for i in 1:SPACEDIM
        for j in 1:SPACEDIM
            for k in 1:SPACEDIM
                rDelta_ULL[i, j, k, :] += 0.5 .* r_gamma_UU[i, :] .* (rhat_D_bar_gamma[j, k, i, :]
                                                                  + rhat_D_bar_gamma[k, j, i, :]
                                                                  - rhat_D_bar_gamma[i, j, k, :])
            end
        end
    end

    rDelta_U = zeros((SPACEDIM, N))
    for i in 1:SPACEDIM
        for j in 1:SPACEDIM
            rDelta_U[i, :] += r_gamma_UU[j, :] .* rDelta_ULL[i, j, j, :]
            for k in 1:SPACEDIM
                rDelta_LLL[i, j, k, :] += r_gamma_LL[i, :] .* rDelta_ULL[i, j, k, :]
            end
        end
    end

    return (rDelta_U, rDelta_ULL, rDelta_LLL)

end

# Ricci tensor in non-coordinate basis
function get_rescaled_ricci_tensor(r, h, dhdr, d2hdr2, lambdar, dlambdardr, rDelta_U, rDelta_ULL, rDelta_LLL, r_gamma_UU, r_gamma_LL)

    N = size(r)[1]
    r_ricci = zeros((SPACEDIM, SPACEDIM, N))

    rhat_D_Lambda = get_rescaled_hat_D_Lambda(r, lambdar, dlambdardr)
    rhat_D2_bar_gamma = get_rescaled_hat_D2_bar_gamma(r, h, dhdr, d2hdr2, r_gamma_UU)

    for i in 1:SPACEDIM
        for j in 1:SPACEDIM
            r_ricci[i, j, :] += (-0.5 * rhat_D2_bar_gamma[i, j, :]
                               .+ 0.5 * (r_gamma_LL[i, :] .* rhat_D_Lambda[j, i, :]
                                      .+ r_gamma_LL[j, :] .* rhat_D_Lambda[i, j, :]))
            for k in 1:SPACEDIM
                r_ricci[i, j, :] += 0.5 * (rDelta_U[k, :] .* rDelta_LLL[i, j, k, :]
                                        .+ rDelta_U[k, :] .* rDelta_LLL[j, i, k, :])
                for m in 1:SPACEDIM
                    r_ricci[i, j, :] += r_gamma_UU[k, :] .* (rDelta_ULL[m, k, i, :] .*
                                                             rDelta_LLL[j, m, k, :]
                                                          .+ rDelta_ULL[m, k, j, :] .*
                                                             rDelta_LLL[i, m, k, :]
                                                          .+ rDelta_ULL[m, i, k, :] .*
                                                             rDelta_LLL[m, j, k, :])

                end
            end
        end
    end

    return r_ricci

end

# \bar{D}_i\Lambda^j but in non-coord basis
function get_rescaled_hat_D_Lambda(r, lambdar, dlambdardr)

    N = size(r)[1]
    rhat_D_Lambda = zeros((SPACEDIM, SPACEDIM, N))

    rflat_chris = get_rescaled_flat_spherical_chris(r)

    rhat_D_Lambda[i_r, i_r, :] = dlambdardr
    for i in 1:SPACEDIM
        for j in 1:SPACEDIM
            rhat_D_Lambda[i, j, :] += rflat_chris[j, i, i_r, :] .* lambdar[:]
        end
    end

    return rhat_D_Lambda

end

# Eq. 27 BMCM (non-coord basis)
function get_rescaled_hat_D2_bar_gamma(r, h, dhdr, d2hdr2, r_gamma_UU)

    N = size(r)[1]
    rhat_D2_bar_gamma = zeros((SPACEDIM, SPACEDIM, N))

    one_over_r = 1.0 ./ r
    rhat_D_bar_gamma = get_rescaled_hat_D_bar_gamma(r, h, dhdr)
    rflat_chris = get_rescaled_flat_spherical_chris(r)

    rhat_D2_bar_gamma[i_r, i_r, :] = r_gamma_UU[i_r, :] .* d2hdr2[i_r, :]
    #rhat_D2_bar_gamma[i_t, i_t, :] = r_gamma_UU[i_r, :] .* (d2hdr2[i_t, :]
    #                                                       + dhdr[i_t, :] .* 2.0 .* one_over_r)
    #rhat_D2_bar_gamma[i_p, i_p, :] = r_gamma_UU[i_r, :] .* (d2hdr2[i_p, :]
    #                                                       + dhdr[i_p, :] .* 2.0 .* one_over_r)
    # original, erroneous
    rhat_D2_bar_gamma[i_t, i_t, :] = r_gamma_UU[i_r, :] .* d2hdr2[i_t, :]
    rhat_D2_bar_gamma[i_p, i_p, :] = r_gamma_UU[i_r, :] .* d2hdr2[i_p, :]
    
    for i in 1:SPACEDIM
        for j in 1:SPACEDIM
            for k in 1:SPACEDIM
                for m in 1:SPACEDIM
                    rhat_D2_bar_gamma[i, j, :] += (-r_gamma_UU[k, :] .*
                                                   (rhat_D_bar_gamma[m, i, j, :] .*
                                                    rflat_chris[m, k, k, :] +
                                                    rhat_D_bar_gamma[k, m, j, :] .*
                                                    rflat_chris[m, i, k, :] +
                                                    rhat_D_bar_gamma[k, i, m, :] .*
                                                    rflat_chris[m, j, k, :]))
                end
            end
        end
    end

    return rhat_D2_bar_gamma

end

# Eq. 25 BMCM in non-coord basis
function get_rescaled_hat_D_bar_gamma(r, h, dhdr)

    N = size(r)[1]
    rhat_D_epsilon = zeros((SPACEDIM, SPACEDIM, SPACEDIM, N))

    r2 = r .* r
    scaling = zeros((3, size(r)[1]))
    scaling[1,:] = ones(size(r)[1])
    scaling[2,:] = r
    scaling[3,:] = r * sintheta

    for i in 1:SPACEDIM
        rhat_D_epsilon[i_r, i, i, :] = dhdr[i, :]
    end

    #rhat_D_epsilon[i_t, i_r, i_r, :] .+= 0.0
    #rhat_D_epsilon[i_t, i_t, i_t, :] .+= 0.0
    #rhat_D_epsilon[i_t, i_p, i_p, :] .+= 0.0

    rhat_D_epsilon[i_t, i_r, i_t, :] .+= (h[i_r, :] - h[i_t, :]) ./ scaling[i_t, :]
    rhat_D_epsilon[i_t, i_t, i_r, :] = rhat_D_epsilon[i_t, i_r, i_t, :]

    #rhat_D_epsilon[i_t, i_r, i_p, :] .+= 0.0
    rhat_D_epsilon[i_t, i_p, i_r, :] = rhat_D_epsilon[i_t, i_r, i_p, :]

    #rhat_D_epsilon[i_p, i_r, i_r, :] .+= 0.0
    #rhat_D_epsilon[i_p, i_t, i_t, :] .+= 0.0
    #rhat_D_epsilon[i_p, i_p, i_p, :] .+= 0.0

    #rhat_D_epsilon[i_p, i_r, i_t, :] .+= 0.0
    rhat_D_epsilon[i_p, i_t, i_r, :] = rhat_D_epsilon[i_p, i_r, i_t, :]

    rhat_D_epsilon[i_p, i_r, i_p, :] .+= (sintheta .* h[i_r, :] .- sintheta .* h[i_p, :]) ./ scaling[i_p, :]
    rhat_D_epsilon[i_p, i_p, i_r, :] = rhat_D_epsilon[i_p, i_r, i_p, :]

    #rhat_D_epsilon[i_p, i_t, i_p, :] .+= 0.0
    rhat_D_epsilon[i_p, i_p ,i_t, :] = rhat_D_epsilon[i_p, i_t, i_p, :]

    return rhat_D_epsilon

end