# TODO: all code here must be checked!!!

function fill_inner_boundary!(state, indices)
    
    for idx in indices
        state[idx, 1:NUM_GHOSTS] .= 
            PARITY[idx] .* state[idx, 2 * NUM_GHOSTS: -1: NUM_GHOSTS + 1]
    end

end

function fill_outer_boundary!(state, indices, r)

    for idx in indices
        b = (state[idx, end-NUM_GHOSTS] - ASYMP_OFFSET[idx]) / 
             r[end-NUM_GHOSTS]^ASYMP_POWER[idx]
        
        bc = ASYMP_OFFSET[idx] .+ b .* r[end-NUM_GHOSTS+1:end].^ASYMP_POWER[idx]
        state[idx, end-NUM_GHOSTS+1:end] .= bc
    end

end

function get_first_derivative(field, indices, drn_matrix, dr)

    dr_field = zeros(size(field))
    for idx in indices
        dr_field[idx, :] = drn_matrix[1, :, :] * field[idx, :] ./ dr
    end

    return dr_field

end

function get_second_derivative(field, indices, drn_matrix, dr)

    dr2_field = zeros(size(field))
    for idx in indices
        dr2_field[idx, :] = drn_matrix[2, :, :] * field[idx, :] ./ dr.^2
    end

    return dr2_field

end

# Note that 0 = false, 1 = true for Julia, so to use logicals as
# array indices I need to pass in the Int-converted logical + 1.
function get_advection(field, direction, indices, advec_x_matrix, dr)

    num_points = size(direction)[1]
    advec_field = zeros(size(field))
    advec_matrix = zeros((num_points,num_points))
    # Build advection matrix
    for i in 1:num_points
        advec_matrix[i,:] = advec_x_matrix[direction[i],i,:]
    end

    for idx in indices
        advec_field[idx, :] = advec_matrix * field[idx, :] ./ dr
    end

    return advec_field

end

function get_kreiss_oliger_diss(state, indices, drn_matrix, dr)

    diss_state = zeros(size(state))
    for idx in indices
        diss_state[idx, :] = drn_matrix[6, :, :] * state[idx, :] ./ (2^6 .* dr)
    end

    return diss_state

end