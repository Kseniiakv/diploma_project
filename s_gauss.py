from scipy import optimize
from calc_tools import *


def hamilt_parts(a, alpha, beta, flavor, A=0.18, B=-1.234):

    mass = quark_masses(flavor)
    mu_q, mu_p = masses(mass)
    b, c = get_params(mass)
    alpha_s = alphas(flavor)

    alpha_matrix = direct_sum(alpha)
    beta_matrix = direct_sum(beta)
    a_dyad = np.tensordot(a, a, axes=0)
    alpha_dyad = np.tensordot(alpha, alpha, axes=0)
    beta_dyad = np.tensordot(beta, beta, axes=0)
    alpha_beta_matrix = alpha_matrix * beta_matrix

    norm_coeff = np.pi**3*np.sum(a_dyad / np.power(alpha_beta_matrix, 3/2))

    kinetic = (3*np.pi**3)*np.sum(a_dyad*(((1/mu_q)*alpha_dyad*beta_matrix +
                                           (1/mu_p)*beta_dyad*alpha_matrix)/
                                           np.power(alpha_beta_matrix, 5/2)))


    summ = 0
    for i in range(len(b)):
        brackets = (b[i]**2)*beta_matrix + (c[i]**2)*alpha_matrix
        pair_potential = np.sum(a_dyad*((0.5*A*brackets - (2/3)*alpha_s[i]*alpha_matrix*beta_matrix)/
                                        ((alpha_matrix*beta_matrix)**2*np.sqrt(brackets))))
        summ += pair_potential

    potential = 2*summ*np.power(np.pi, 5/2)
    return (kinetic+potential)/norm_coeff + sum(mass) + B


def hamiltonian(params, flavor):
    n = int((len(params)+1)/3)
    a = [1]
    a.extend(params[:(n - 1)])
    a = np.array(a)
    alpha = np.array(params[(n-1):(2*n-1)])
    beta = np.array(params[(2*n-1):(3*n-1)])
    hamilt = hamilt_parts(a, alpha, beta, flavor)
    return hamilt


def spin_potential(params, flavor, spin):

    n = int((len(params) + 1) / 3)
    a = [1]
    a.extend(params[:(n - 1)])
    a = np.array(a)
    alpha = np.array(params[(n - 1):(2 * n - 1)])
    beta = np.array(params[(2 * n - 1):(3 * n - 1)])

    mass = quark_masses(flavor)
    b, c = get_params(mass)
    alpha_s = alphas(flavor)
    s_pair = mean_spin(spin)

    alpha_matrix = direct_sum(alpha)
    beta_matrix = direct_sum(beta)
    a_dyad = np.tensordot(a, a, axes=0)
    alpha_beta_matrix = alpha_matrix * beta_matrix

    norm_coeff = np.pi ** 3 * np.sum(a_dyad / np.power(alpha_beta_matrix, 3 / 2))

    prod_mass = np.array([mass[i] * mass[j] for i in range(len(mass))
                             for j in range(len(mass)) if i != j and i < j])

    summ = 0
    for i in range(len(b)):
        brackets = np.power((b[i] ** 2) * beta_matrix + (c[i] ** 2) * alpha_matrix, 1.5)
        pair_sspotential = alpha_s[i]*s_pair[i]*np.sum(a_dyad / brackets) / prod_mass[i]
        summ += pair_sspotential

    ss_potential = (16/9) * summ * np.power(np.pi, 5 / 2)
    return ss_potential/norm_coeff


def one_particle_mass(flavor, spin, B=-1.234):
    Initial = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    bounds = [(None, None), (None, None), (0.00000001, None), (0.00000001, None),
              (0.00000001, None), (0.00000001, None), (0.00000001, None), (0.00000001, None)]

    res = optimize.minimize(hamiltonian, Initial, args=(flavor), method='SLSQP', bounds=bounds)
    ss_potential = spin_potential(res.x, flavor, spin)

    total_mass = res.fun + ss_potential
    mass = quark_masses(flavor)
    pure_t_v = res.fun - sum(mass) - B
    return pure_t_v, ss_potential, total_mass


def light_spectre():

    proton = one_particle_mass(np.array([0, 0, 0]), (0.5, 0))
    lambda_p = one_particle_mass(np.array([1, 0, 0]), (0.5, 0))
    sigma_plus = one_particle_mass(np.array([1, 0, 0]), (0.5, 1))
    xi_minus = one_particle_mass(np.array([1, 1, 0]), (0.5, 0))
    delta = one_particle_mass(np.array([0, 0, 0]), (1.5, 1))
    sigma_star = one_particle_mass(np.array([1, 0, 0]), (1.5, 1))
    xi_star = one_particle_mass(np.array([1, 1, 0]), (1.5, 1))
    omega = one_particle_mass(np.array([1, 1, 1]), (1.5, 1))

    table = [0.938, 1.116, 1.189, 1.322, 1.232, 1.383, 1.532, 1.672]
    spectre = [proton, lambda_p, sigma_plus, xi_minus, delta, sigma_star, xi_star, omega]

    for i in range(len(spectre)):
        print(spectre[i], (spectre[i][-1] - table[i]))
#light_spectre()


def c_spectre():

    lambda_c = one_particle_mass(np.array([2, 0, 0]), (0.5, 0))
    sigma_c = one_particle_mass(np.array([2, 0, 0]), (0.5, 1))
    sigma_c_star = one_particle_mass(np.array([2, 0, 0]), (1.5, 1))
    xi_c = one_particle_mass(np.array([2, 0, 1]), (0.5, 0))
    xi_c_prime = one_particle_mass(np.array([2, 0, 1]), (0.5, 1))
    xi_c_star = one_particle_mass(np.array([2, 0, 1]), (1.5, 1))
    omega_c = one_particle_mass(np.array([2, 1, 1]), (0.5, 1))
    omega_c_star = one_particle_mass(np.array([2, 1, 1]), (1.5, 1))

    table = [2.286, 2.454, 2.518, 2.468, 2.576, 2.646, 2.695, 2.766]
    spectre = [lambda_c, sigma_c, sigma_c_star, xi_c, xi_c_prime, xi_c_star, omega_c, omega_c_star]

    for i in range(len(spectre)):
        print(spectre[i], (spectre[i][-1] - table[i]))
c_spectre()


def b_spectre():

    lambda_b = one_particle_mass(np.array([3, 0, 0]), (0.5, 0))
    sigma_b = one_particle_mass(np.array([3, 0, 0]), (0.5, 1))
    sigma_b_star = one_particle_mass(np.array([3, 0, 0]), (1.5, 1))
    xi_b = one_particle_mass(np.array([3, 0, 1]), (0.5, 0))
    xi_b_prime = one_particle_mass(np.array([3, 0, 1]), (0.5, 1))
    xi_b_star = one_particle_mass(np.array([3, 0, 1]), (1.5, 1))
    omega_b = one_particle_mass(np.array([3, 1, 1]), (0.5, 1))
    omega_b_star = one_particle_mass(np.array([3, 1, 1]), (1.5, 1))

    table = [5.620, 5.811, 5.832, 5.795, 66666, 5.949, 6.049, 66666]
    spectre = [lambda_b, sigma_b, sigma_b_star, xi_b, xi_b_prime, xi_b_star, omega_b, omega_b_star]

    for i in range(len(spectre)):
        print(spectre[i], (spectre[i][-1] - table[i]))
# b_spectre()


def double_heavy_spectre():
    xi_cc = one_particle_mass(np.array([0, 2, 2]), (0.5, 1))
    xi_cc_star = one_particle_mass(np.array([0, 2, 2]), (1.5, 1))
    xi_bb = one_particle_mass(np.array([0, 3, 3]), (0.5, 1))
    xi_bb_star = one_particle_mass(np.array([0, 3, 3]), (1.5, 1))

    omega_cc = one_particle_mass(np.array([1, 2, 2]), (0.5, 1))
    omega_cc_star = one_particle_mass(np.array([1, 2, 2]), (1.5, 1))
    omega_bb = one_particle_mass(np.array([1, 3, 3]), (0.5, 1))
    omega_bb_star = one_particle_mass(np.array([1, 3, 3]), (1.5, 1))

    spectre = [xi_cc, xi_cc_star, xi_bb, xi_bb_star, omega_cc, omega_cc_star, omega_bb, omega_bb_star]

    for particle in spectre:
        print(particle)
#double_heavy_spectre()

#print(one_particle_mass(([1,0,0]), ((0.5, 0))))
