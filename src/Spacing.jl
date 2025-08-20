using Roots

function get_spacing_parameters(r_max, min_dr, max_dr, spacing_type)
    if spacing_type == "Linear"
        x_max = r_max
        dx = min_dr
        num_points = ceil(x_max / dx + NUM_GHOSTS + 1 / 2)

        parameters = Dict(:r_max => r_max, :num_points => Int(num_points), :x_max => x_max, :a => 1.0, :spacing_type => "Linear")
        return parameters

    elseif spacing_type == "Sinh"
        f(x) = tanh(x) - sinh(x) * min_dr / max_dr
        x_max = find_zero(f, (min_dr / max_dr, max_dr / min_dr))
        a = r_max / sinh(x_max)
        dx = min_dr / a
        num_points = ceil(x_max / dx + NUM_GHOSTS + 1 / 2)

        parameters = Dict(:r_max => r_max, :num_points => Int(num_points), :x_max => x_max, :a => a, :spacing_type => "Sinh")
        return parameters

    elseif spacing_type == "Cubic"
        x_max = sqrt(max_dr / min_dr - 1)
        a = r_max / (x_max + x_max^3 / 3)
        dx = min_dr / a
        num_points = ceil(x_max / dx + NUM_GHOSTS + 1 / 2)

        parameters = Dict(:r_max => r_max, :num_points => Int(num_points), :x_max => x_max, :a => a, :spacing_type => "Cubic")
        return parameters
    end
end

function get_spacing(params)

    num_points = params[:num_points]
    x_max = params[:x_max]
    a = params[:a]
    spacing_type = params[:spacing_type]

    all_points = 2 * (num_points - NUM_GHOSTS)
    x = collect(LinRange(-x_max, x_max, all_points))
    dx = x[2]-x[1]
    x = x[end-num_points+1:end]

    dnr_dxn = zeros((6, num_points))

    if (spacing_type == "Linear")
        r = x
        dnr_dxn[1,:] = ones(num_points)
    elseif (spacing_type == "Sinh")
        r = a .* sinh.(x)
        dnr_dxn[2,:] = a .* sinh.(x)
        dnr_dxn[4,:] = a .* sinh.(x)
        dnr_dxn[6,:] = a .* sinh.(x)
        dnr_dxn[1,:] = a .* cosh.(x)
        dnr_dxn[3,:] = a .* cosh.(x)
        dnr_dxn[5,:] = a .* cosh.(x)
    elseif (spacing_type == "Cubic")
        r = a .* (x .+ x.^3 ./ 3)
        dnr_dxn[1,:] = a .* (1 .+ x.^2)
        dnr_dxn[2,:] = 2 .* a .* x
        dnr_dxn[3,:] .= 2 .* a
    end

    dr = dnr_dxn[1,:] * dx;

    return (r, dr, dx, dnr_dxn)
end
