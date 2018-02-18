import numpy as np

def quark_masses(flavor):
    m = [0.335, 0.65, 1.92, 5.3]
    return np.array([m[i] for i in flavor])


def n_f_number(flavor):
    if min(flavor) == 0 and (max(flavor) == 0 or max(flavor) == 1): n_f_num = 2
    elif min(flavor) == 0 and (max(flavor) == 2 or max(flavor) == 3): n_f_num = 3
    elif min(flavor) == 1: n_f_num = 3
    elif min(flavor) == 2: n_f_num = 4
    else: n_f_num = 5
    return n_f_num


def masses(m):
    """returns reduced masses of the system"""
    M = sum(m)
    mu_q = 3*m[0] * (m[1] + m[2]) / (4 * M)
    mu_p = m[1]*m[2] / (m[1] + m[2])
    return mu_q, mu_p


def get_params(m):
    b_1 = np.sqrt(3)/2
    b = [b_1, b_1, 0]
    c_1 = -m[2]/(m[1] + m[2])
    c_2 = m[1]/(m[1] + m[2])
    c_3 = 1
    c = [c_1, c_2, c_3]
    return np.array(b), np.array(c)


def direct_sum(vector):
    n = len(vector)
    vector1 = vector.reshape(1, n)
    vector2 = vector.reshape(n, 1)
    return vector1 + vector2


def mean_spin(spin):
    J, s = spin
    s_23 = 0.5 * (s * (s + 1) - 1.5)
    s_12 = 0.25 * (J * (J + 1) - s * (s + 1) - 0.75)
    return np.array([s_12, s_12, s_23])


def alphas(flavor, m_0=0.95, lambda_qcd=0.413):
    m = quark_masses(flavor)
    flav_pair = [(flavor[i], flavor[j]) for i in range(len(flavor))
                         for j in range(len(flavor)) if i != j and i < j]

    reduced_mass = np.array([2*m[i]*m[j]/(m[i]+m[j]) for i in range(len(m))
                         for j in range(len(m)) if i != j and i < j])

    n_f = np.array([n_f_number(pair) for pair in flav_pair])
    beta_0 = 11 - (2/3)*n_f

    alpha = 4*np.pi/(beta_0*np.log((reduced_mass**2 + m_0**2)/lambda_qcd**2))

    return alpha