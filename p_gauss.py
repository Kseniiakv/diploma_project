from scipy import optimize, special
from calc_tools import *
import pandas as pd


def hamilt_parts(a, alpha, beta, flavor, orb_moment, A=0.18, B=-1.234):

    mass = quark_masses(flavor)
    mu_q, mu_p = masses(mass)
    b, c = get_params(mass)
    alpha_s = alphas(flavor)
    L_q, l_p = orb_moment

    alpha_matrix = direct_sum(alpha)
    beta_matrix = direct_sum(beta)
    a_dyad = np.tensordot(a, a, axes=0)
    alpha_dyad = np.tensordot(alpha, alpha, axes=0)
    beta_dyad = np.tensordot(beta, beta, axes=0)
    alpha_beta_matrix = alpha_matrix * beta_matrix

    gammas = special.gamma(L_q+1.5)*special.gamma(l_p+1.5)

    norm_sum = np.sum(a_dyad / (np.power(alpha_matrix, (L_q+1.5)) *
                                np.power(beta_matrix, (l_p+1.5))))
    norm_coeff = 4*np.pi**2*gammas*norm_sum

    kinetic_sum = np.sum(a_dyad*(((1/mu_q)*(L_q + 1.5)*alpha_dyad*beta_matrix +
                                           (1/mu_p)*(l_p + 1.5)*beta_dyad*alpha_matrix)/
                    (np.power(alpha_matrix, (L_q+2.5))*np.power(beta_matrix, (l_p+2.5)))))

    kinetic = 8*np.pi**2*gammas*kinetic_sum

    conf_potential = 0.5 * A * 4 * np.pi**2 * \
        np.sum(pair_potential(1, a_dyad, alpha_matrix, beta_matrix, b[i], c[i], orb_moment)
                                for i in range(len(b)))
    coul_potential = (-2/3) * 4 * np.pi**2 * \
        np.sum(alpha_s[i] *
            pair_potential(-1, a_dyad, alpha_matrix, beta_matrix, b[i], c[i], orb_moment)
                                for i in range(len(b)))

    return (kinetic + conf_potential + coul_potential)/norm_coeff + sum(mass) + B


def pair_potential(n, a_dyad, alpha_matrix, beta_matrix, b_ij, c_ij, orb_moment):

    L_q, l_p = orb_moment
    summ_k = 0
    for k in range(0, 2*l_p+1, 2):
        summ_p = 0
        for p in range(0, 2*L_q+1, 2):
            gammas = special.gamma((2 * (L_q + l_p) - k - p + 3) / 2) * \
                     special.gamma((n + k + p + 3) / 2)

            bc = c_ij**(2*L_q + k - p) * b_ij**(2*l_p - k + p)
            alpha_beta = np.power(alpha_matrix, (n + 3 + p - k) / 2) * \
                         np.power(beta_matrix, (n + 3 - p + k) / 2)
            summ_p += (special.binom(2*L_q, p) * gammas * bc / alpha_beta)
        summ_k += (special.binom(2*l_p, k)*summ_p)
    brackets = (b_ij ** 2) * beta_matrix + (c_ij ** 2) * alpha_matrix

    return np.sum(a_dyad * np.power(brackets, (n/2 - l_p - L_q)) * summ_k)


def hamiltonian(params, flavor, orb_moment):
    n = int((len(params)+1)/3)
    a = [1]
    a.extend(params[:(n - 1)])
    a = np.array(a)
    alpha = np.array(params[(n-1):(2*n-1)])
    beta = np.array(params[(2*n-1):(3*n-1)])
    hamilt = hamilt_parts(a, alpha, beta, flavor, orb_moment)
    return hamilt


def spin_potential(params, flavor, spin, orb_moment=(0, 0)):

    n = int((len(params) + 1) / 3)
    a = [1]
    a.extend(params[:(n - 1)])
    a = np.array(a)
    alpha = np.array(params[(n - 1):(2 * n - 1)])
    beta = np.array(params[(2 * n - 1):(3 * n - 1)])
    L_q, l_p = orb_moment

    mass = quark_masses(flavor)
    b, c = get_params(mass)
    alpha_s = alphas(flavor)
    s_pair = mean_spin(spin)

    alpha_matrix = direct_sum(alpha)
    beta_matrix = direct_sum(beta)
    a_dyad = np.tensordot(a, a, axes=0)
    alpha_beta_matrix = alpha_matrix * beta_matrix

    gammas = special.gamma(L_q + 1.5) * special.gamma(l_p + 1.5)

    norm_sum = np.sum(a_dyad / (np.power(alpha_matrix, (L_q + 1.5)) *
                                np.power(beta_matrix, (l_p + 1.5))))
    norm_coeff = 4 * np.pi ** 2 * gammas * norm_sum

    prod_mass = np.array([mass[i] * mass[j] for i in range(len(mass))
                             for j in range(len(mass)) if i != j and i < j])

    summ = 0
    for i in range(len(b)):
        brackets = np.power((b[i] ** 2) * beta_matrix + (c[i] ** 2) *
                            alpha_matrix, (L_q + l_p) + 1.5)
        pair_sspotential = alpha_s[i]*s_pair[i] * c[i]**(2*L_q) * b[i]**(2 * l_p) *\
                           np.sum(a_dyad / brackets) / prod_mass[i]
        summ += pair_sspotential

    ss_potential = (32/9) * summ * np.power(np.pi, 2) * special.gamma((L_q + l_p) + 1.5)
    return ss_potential/norm_coeff


def one_particle_mass(flavor, spin, orb_moment=(0, 0), B=-1.234):
    Initial = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    bounds = [(None, None), (None, None), (0.00000001, None), (0.00000001, None),
              (0.00000001, None), (0.00000001, None), (0.00000001, None), (0.00000001, None)]

    res = optimize.minimize(hamiltonian, Initial, args=(flavor, orb_moment), method='SLSQP', bounds=bounds)
    ss_potential = spin_potential(res.x, flavor, spin)

    total_mass = res.fun + ss_potential
    mass = quark_masses(flavor)
    pure_t_v = res.fun - sum(mass) - B
    return total_mass, pure_t_v, ss_potential


def calculate(spectre):
    values = {}
    for particle in spectre:
        flavor = np.array(spectre[particle]['flavor'])
        spin = spectre[particle]['spin']
        name = spectre[particle]['name']
        exp = spectre[particle]['exp']
        total_mass, pure_t_v, ss_potential = one_particle_mass(flavor, spin)
        total_mass *= 1000
        pure_t_v *=1000
        ss_potential *= 1000
        exp *= 1000
        values[name] = {'Total mass': total_mass, 'T + V': pure_t_v,
                        'ss potential': ss_potential, 'Difference': total_mass - exp}
    return values


light_spectre = {'proton': {'flavor': [0, 0, 0], 'spin': (0.5, 0), 'exp': 0.938, 'name': 'p  1/2(1/2)'},
                 'lambda': {'flavor': [1, 0, 0], 'spin': (0.5, 0), 'exp': 1.116, 'name': u'\u039B  0(1/2)'},
             'sigma_plus': {'flavor': [1, 0, 0], 'spin': (0.5, 1), 'exp': 1.189, 'name': u'\u03A3+ 1(1/2)'},
               'xi_minus': {'flavor': [1, 1, 0], 'spin': (0.5, 0), 'exp': 1.322, 'name': u'\u039E- 1/2(1/2)'},
                  'delta': {'flavor': [0, 0, 0], 'spin': (1.5, 1), 'exp': 1.232, 'name': u'\u0394  3/2(3/2)'},
             'sigma_star': {'flavor': [1, 0, 0], 'spin': (1.5, 1), 'exp': 1.383, 'name': u'\u03A3* 1(3/2)'},
                'xi_star': {'flavor': [1, 1, 0], 'spin': (1.5, 1), 'exp': 1.532, 'name': u'\u039E* 1/2(3/2)'},
                  'omega': {'flavor': [1, 1, 1], 'spin': (1.5, 1), 'exp': 1.672, 'name': u'\u03A9  0(3/2)'},
                }


c_spectre = {'lambda_c': {'flavor': [2, 0, 0], 'spin': (0.5, 0), 'exp': 2.286, 'name': u'\u039Bc 0(1/2)'},
              'sigma_c': {'flavor': [2, 0, 0], 'spin': (0.5, 1), 'exp': 2.454, 'name': u'\u03A3c 1(1/2)'},
         'sigma_c_star': {'flavor': [2, 0, 0], 'spin': (1.5, 1), 'exp': 2.518, 'name': u'\u03A3c 1(3/2)'},
                 'xi_c': {'flavor': [2, 0, 1], 'spin': (0.5, 0), 'exp': 2.468, 'name': u'\u039Ec 1/2(1/2)'},
           'xi_c_prime': {'flavor': [2, 0, 1], 'spin': (0.5, 1), 'exp': 2.576, 'name': u"\u039Ec'1/2(1/2)"},
            'xi_c_star': {'flavor': [2, 0, 1], 'spin': (1.5, 1), 'exp': 2.646, 'name': u'\u039Ec 1/2(3/2)'},
              'omega_c': {'flavor': [2, 1, 1], 'spin': (0.5, 1), 'exp': 2.695, 'name': u'\u03A9c 0(1/2)'},
         'omega_c_star': {'flavor': [2, 1, 1], 'spin': (1.5, 1), 'exp': 2.766, 'name': u'\u03A9c 0(3/2)'},
                }

b_spectre = {'lambda_b': {'flavor': [3, 0, 0], 'spin': (0.5, 0), 'exp': 5.620, 'name': u'\u039Bb 0(1/2)'},
              'sigma_b': {'flavor': [3, 0, 0], 'spin': (0.5, 1), 'exp': 5.811, 'name': u'\u03A3b 1(1/2)'},
         'sigma_b_star': {'flavor': [3, 0, 0], 'spin': (1.5, 1), 'exp': 5.832, 'name': u'\u03A3b 1(3/2)'},
                 'xi_b': {'flavor': [3, 0, 1], 'spin': (0.5, 0), 'exp': 5.795, 'name': u'\u039Eb 1/2(1/2)'},
           'xi_b_prime': {'flavor': [3, 0, 1], 'spin': (0.5, 1), 'exp': 66666, 'name': u"\u039Eb'1/2(1/2)"},
            'xi_b_star': {'flavor': [3, 0, 1], 'spin': (1.5, 1), 'exp': 5.949, 'name': u'\u039Eb 1/2(3/2)'},
              'omega_b': {'flavor': [3, 1, 1], 'spin': (0.5, 1), 'exp': 6.049, 'name': u'\u03A9b 0(1/2)'},
         'omega_b_star': {'flavor': [3, 1, 1], 'spin': (1.5, 1), 'exp': 66666, 'name': u'\u03A9b 0(3/2)'},
                }

double_spectre = {'xi_cc': {'flavor': [0, 2, 2], 'spin': (0.5, 1), 'exp': 3.685, 'name': u'\u039Ecc 1/2(1/2)'},
             'xi_cc_star': {'flavor': [0, 2, 2], 'spin': (1.5, 1), 'exp': 3.754, 'name': u'\u039Ecc 1/2(3/2)'},
                 'xi_bbr': {'flavor': [0, 3, 3], 'spin': (0.5, 1), 'exp': 10.314, 'name': u'\u039Ebb 1/2(1/2)'},
             'xi_bb_star': {'flavor': [0, 3, 3], 'spin': (1.5, 1), 'exp': 10.339, 'name': u'\u039Ebb 1/2(3/2)'},
               'omega_cc': {'flavor': [1, 2, 2], 'spin': (0.5, 1), 'exp': 3.832, 'name': u"\u03A9cc 1/2(1/2)"},
          'omega_cc_star': {'flavor': [1, 2, 2], 'spin': (1.5, 1), 'exp': 3.883, 'name': u'\u03A9cc 1/2(3/2)'},
               'omega_bb': {'flavor': [1, 3, 3], 'spin': (0.5, 1), 'exp': 10.454, 'name': u'\u03A9bb 1/2(1/2)'},
          'omega_bb_star': {'flavor': [1, 3, 3], 'spin': (1.5, 1), 'exp': 10.486, 'name': u'\u03A9bb 1/2(3/2)'},
                 }

light = pd.DataFrame(calculate(light_spectre)).T
# light.to_csv('light_spectre10.csv', sep='\t', float_format='%.2f')
print(light)

# c_heavy = pd.DataFrame(calculate(c_spectre)).T
# c_heavy.to_csv('c_heavy10.csv', sep='\t', float_format='%.2f')
# print(c_heavy)

# b_heavy = pd.DataFrame(calculate(b_spectre)).T
# b_heavy.to_csv('b_heavy10.csv', sep='\t', float_format='%.2f')
# print(b_heavy)

# double_heavy = pd.DataFrame(calculate(double_spectre)).T
# double_heavy.to_csv('double_heavy10.csv', sep='\t', float_format='%.2f')
# print(double_heavy)





