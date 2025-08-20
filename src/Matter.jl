function V_of_u(u)
    return 0.5 * scalar_mu * scalar_mu .* u .* u
end

function dVdu(u)
    return scalar_mu * scalar_mu .* u
end

function get_matter_rhs(u, v, dudr, d2udr2, r_gamma_UU, em4phi, dphidr, K, lapse, dlapsedr, r_conformal_chris, dhdr)

    dudt = lapse .* v
    dvdt = lapse .* K .* v .+ r_gamma_UU[i_r, :] .* em4phi .* (2.0 .* lapse .* dphidr .* dudr
                                                              .+ lapse .* d2udr2
                                                              .+ dlapsedr .* dudr)
    
    #for i in 1:SPACEDIM
        #dvdt .+= -em4phi .* lapse .* r_gamma_UU[i_r, :] .* r_conformal_chris[i_r, i, i, :] .* dudr
    dvdt .+= 0.5 .* em4phi .* r_gamma_UU[i_r, :] .* lapse .* dudr .* (-r_gamma_UU[i_r, :] .* dhdr[i_r, :])
    dvdt .+= 0.5 .* em4phi .* r_gamma_UU[i_r, :] .* lapse .* dudr .* (r_gamma_UU[i_t, :] .* dhdr[i_t, :])
    dvdt .+= 0.5 .* em4phi .* r_gamma_UU[i_r, :] .* lapse .* dudr .* (r_gamma_UU[i_p, :] .* dhdr[i_p, :])
    #end

    dvdt .+= -lapse .* dVdu(u)

    return (dudt, dvdt)
    
end

function get_rho(u, dudr, v, r_gamma_UU, em4phi)

    return 0.5 .* v .* v .+ 0.5 .* em4phi .* r_gamma_UU[i_r, :] .* dudr .* dudr .+ V_of_u(u)
    
end

function get_Si(u, dudr, v)

    N = size(u)[1]
    S_i = zeros((SPACEDIM, N))

    S_i[i_r, :] = -v .* dudr

    return S_i

end

function get_rescaled_Sij(u, dudr, v, r_gamma_UU, em4phi, r_gamma_LL)

    N = size(u)[1]
    rS_ij = zeros((SPACEDIM, SPACEDIM, N))

    Vt = -v .* v .+ em4phi .* r_gamma_UU[i_r, :] .* dudr .* dudr
    for i in 1:SPACEDIM
        rS_ij[i, i, :] = -(0.5 .* Vt .+ V_of_u(u)) .* r_gamma_LL[i, :] ./ em4phi .+ delta[i, i_r] .* dudr .* dudr
    end

    S = zeros((N))
    for i in 1:SPACEDIM
        S += rS_ij[i, i ,:] .* r_gamma_UU[i, :] .* em4phi
    end

    return (S, rS_ij)

end