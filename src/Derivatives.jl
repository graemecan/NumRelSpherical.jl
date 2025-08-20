function get_dxn_matrix(num_points)
    dxn_matrix = zeros((6, num_points, num_points))

    identity_u3 = diagm( 3=>repeat([1.0], num_points-3))
    identity_u2 = diagm( 2=>repeat([1.0], num_points-2))
    identity_u1 = diagm( 1=>repeat([1.0], num_points-1))
    identity =    diagm( 0=>repeat([1.0], num_points))
    identity_l1 = diagm(-1=>repeat([1.0], num_points-1))
    identity_l2 = diagm(-2=>repeat([1.0], num_points-2))
    identity_l3 = diagm(-3=>repeat([1.0], num_points-3))

    dxn_matrix[1,:,:] = ((1/12)*identity_l2 - ( 2/3)*identity_l1 
                       + ( 2/3)*identity_u1 - (1/12)*identity_u2)

    dxn_matrix[2,:,:] = (-(1/12)*identity_l2 + ( 4/3)*identity_l1 
                         -( 5/2)*identity + ( 4/3)*identity_u1 
                         -(1/12)*identity_u2)

    dxn_matrix[3,:,:] = (-(1/2)*identity_l2 + identity_l1
                         -identity_u1 + (1/2)*identity_u2)

    dxn_matrix[4,:,:] = (identity_l2 - 4*identity_l1 + 6*identity
                       - 4*identity_u1 + identity_u2)

    dxn_matrix[5,:,:] = (-(1/2)*identity_l3 + 2*identity_l2 - (5/2)*identity_l1
                        + (5/2)*identity_u1 - 2*identity_u2 + (1/2)*identity_u3)

    dxn_matrix[6,:,:] = (identity_l3 - 6*identity_l2 + 15*identity_l1
                    - 20*identity + 15*identity_u1 - 6*identity_u2 + identity_u3)

    dxn_matrix[:,1:NUM_GHOSTS,:] .= 0.0
    dxn_matrix[:,end-NUM_GHOSTS+1:end,:] .= 0.0

    return dxn_matrix
end

function get_drn_matrix(dx, dnr_dxn, num_points, dxn_matrix)

    drn_matrix = zeros((7, num_points, num_points))

    dr_dx = dnr_dxn[1,:]

    drn_matrix[1,:,:] = dxn_matrix[1,:,:]
    drn_matrix[2,:,:] = (dxn_matrix[2,:,:] 
            - dx * Diagonal(dnr_dxn[2,:]./dr_dx) * drn_matrix[1,:,:])
    drn_matrix[3,:,:] = (dxn_matrix[3,:,:] 
            - dx * Diagonal(3 * dnr_dxn[2,:]./dr_dx) * drn_matrix[2,:,:]
            - dx^2 * Diagonal(dnr_dxn[3,:]./dr_dx) * drn_matrix[1,:,:])
    drn_matrix[4,:,:] = (dxn_matrix[4,:,:]
            - dx * Diagonal(6 * dnr_dxn[2,:]./dr_dx) * drn_matrix[3,:,:]
            - dx^2 * Diagonal(4 * dnr_dxn[3,:]./dr_dx + 3 * (dnr_dxn[2,:]./dr_dx).^2) * drn_matrix[2,:,:]
            - dx^3 * Diagonal(dnr_dxn[4,:]./dr_dx) * drn_matrix[1,:,:])
    drn_matrix[5,:,:] = (dxn_matrix[5,:,:]
            - dx * Diagonal(10 * dnr_dxn[2,:]./dr_dx) * drn_matrix[4,:,:]
            - dx^2 * Diagonal(10 * dnr_dxn[3,:]./dr_dx + 15 * (dnr_dxn[2,:]./dr_dx).^2) * drn_matrix[3,:,:]
            - dx^3 * Diagonal(10 * dnr_dxn[3,:] .* dnr_dxn[2,:]./dr_dx.^2 + 5 * dnr_dxn[4,:] ./ dr_dx) * drn_matrix[2,:,:]
            - dx^4 * Diagonal(dnr_dxn[5,:]./dr_dx) * drn_matrix[1,:,:])
    drn_matrix[6,:,:] = (dxn_matrix[6,:,:]
            - dx * Diagonal(15 * dnr_dxn[2,:]./dr_dx) * drn_matrix[5,:,:]
            - dx^2 * Diagonal(45 * (dnr_dxn[2,:]./dr_dx).^2 + 20 * dnr_dxn[3,:]./dr_dx) * drn_matrix[4,:,:]
            - dx^3 * Diagonal(15 * dnr_dxn[4,:]./dr_dx + 60 * dnr_dxn[3,:] .* dnr_dxn[2,:]./dr_dx.^2 + 15 * (dnr_dxn[2,:]./dr_dx).^3) * drn_matrix[3,:,:]
            - dx^4 * Diagonal(6 * dnr_dxn[5,:]./dr_dx + 15 * dnr_dxn[4,:] .* dnr_dxn[2,:] ./ dr_dx.^2 + 10 * (dnr_dxn[3,:]./dr_dx).^2) * drn_matrix[2,:,:]
            - dx^5 * Diagonal(dnr_dxn[6,:]./dr_dx) * drn_matrix[1,:,:])

    return drn_matrix
end

function compute_advec_x_matrix(num_points)

    identity_u3 = diagm( 3=>repeat([1.0], num_points-3))
    identity_u2 = diagm( 2=>repeat([1.0], num_points-2))
    identity_u1 = diagm( 1=>repeat([1.0], num_points-1))
    identity =    diagm( 0=>repeat([1.0], num_points))
    identity_l1 = diagm(-1=>repeat([1.0], num_points-1))
    identity_l2 = diagm(-2=>repeat([1.0], num_points-2))
    identity_l3 = diagm(-3=>repeat([1.0], num_points-3))

    advec_x_matrix = zeros((2,num_points,num_points))

    advec_x_matrix[1,:,:] = ((-1/3)*identity_l3
                            + (3/2)*identity_l2
                            - 3*identity_l1
                            + (11/6)*identity)

    advec_x_matrix[2,:,:] = ((-11/6)*identity
                            + 3*identity_u1
                            - (3/2)*identity_u2
                            + (1/3)*identity_u3)

    advec_x_matrix[1, 1:NUM_GHOSTS, :] .= 0.0
    advec_x_matrix[2, end-NUM_GHOSTS+1:end, :] .= 0.0

    return advec_x_matrix

end
