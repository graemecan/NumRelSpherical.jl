module NumRelSpherical

using LinearAlgebra

include("./Parameters.jl")
include("./Grid.jl")
include("./Spacing.jl")
include("./Derivatives.jl")
include("./Tensors.jl")
include("./Matter.jl")
include("./Bssnrhs.jl")
include("./Rhs.jl")
include("./InitialConditions.jl")

end # module NumRelSpherical
