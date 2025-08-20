const NUM_GHOSTS = 3
const NUM_VARS = 14
const SPACEDIM = 3

const i_r, i_t, i_p = 1, 2, 3

const idx_u = 1
const idx_v = 2
const idx_phi = 3
const idx_hrr = 4
const idx_htt = 5
const idx_hpp = 6
const idx_K = 7
const idx_arr = 8
const idx_att = 9
const idx_app = 10
const idx_lambdar = 11
const idx_shiftr = 12
const idx_br = 13
const idx_lapse = 14

const ALL_INDICES = [idx_u, idx_v, idx_phi, idx_hrr, idx_htt,
                     idx_hpp, idx_K, idx_arr, idx_att, idx_app,
                     idx_lambdar, idx_shiftr, idx_br, idx_lapse]

const PARITY = [1, 1,
                1, 1, 1, 1,
                1, 1, 1, 1,
                -1, -1, -1, 1]

const ASYMP_POWER = [0., 0.,
                     -1., -1., -1., -1.,
                     -1., -2., -2., -2.,
                     -2., -1., -1., 0.]

const ASYMP_OFFSET = [0., 0.,
                      0., 0., 0., 0.,
                      0., 0., 0., 0.,
                      0., 0., 0., 1.]

const sintheta = 1
const sin2theta = 1
const costheta = 0
const cos2theta = 0

const one_sixth = 1/6
const one_third = 1/3
const two_thirds = 2/3
const four_thirds = 4/3

const eight_pi_G = 8 * π #G=c=1

const scalar_mu = 1.0

const delta = Matrix{Float64}(I, 3, 3)