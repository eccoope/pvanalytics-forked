
# Get C_V, C_T

# Get Transmission, voltage ratio


import numpy as np

def _a_func(module_temp, imp0, c1, alpha_imp, t_ref=25):
    return imp0*c1*(1 + alpha_imp*(module_temp - t_ref))
def _b_func(module_temp, imp0, c0, alpha_imp, t_ref=25):
    return imp0*c0*(1 + alpha_imp*(module_temp - t_ref))
def _c_func(current):
    return -current

def _solve_quadratic_eqn(module_temp, current, imp0, c0, c1, alpha_imp, t_ref=25):
    discriminant = (_b_func(module_temp,
                            module_temp,
                            imp0, c0,
                            alpha_imp))**2 - 4*_a_func(module_temp,
                                                       imp0, c0,
                                                       alpha_imp)*_c_func(current)
    sol1 = (-_b_func(module_temp) + np.sqrt(discriminant))/(2*_a_func(module_temp))
    sol2 = (-_b_func(module_temp) - np.sqrt(discriminant))/(2*_a_func(module_temp))
    return sol1, sol2


def get_transmission(module_temp, current, E_e, scaling_factor, imp0, c0,
                     c1, alpha_imp, t_ref):
    T1, T2 = _solve_quadratic_eqn(current.to_numpy()/scaling_factor,
                                  module_temp.to_numpy(),
                                  imp0,
                                  c0,
                                  c1,
                                  alpha_imp)/(E_e.to_numpy())
    
    T1[np.argwhere(current.to_numpy() == 0)] = 0
    T2[np.argwhere(current.to_numpy() == 0)] = 0

    T1[np.argwhere(np.isnan(current.to_numpy()))] = np.nan
    T2[np.argwhere(np.isnan(current.to_numpy()))] = np.nan

    T1[np.argwhere(T1 < 0)] = np.nan
    T2[np.argwhere(T2 < 0)] = np.nan

    return T1, T2

def model_v(T, E_e, module_temperature)