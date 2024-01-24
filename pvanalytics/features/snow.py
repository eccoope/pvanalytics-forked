
import numpy as np
from scipy.constants import Boltzmann, elementary_charge

def _a_func(cell_temp, imp0, c1, alpha_imp, t_ref=25):
    return imp0*c1*(1 + alpha_imp*(cell_temp - t_ref))

def _b_func(cell_temp, imp0, c0, alpha_imp, t_ref=25):
    return imp0*c0*(1 + alpha_imp*(cell_temp - t_ref))

def _c_func(current):
    return -current

def _solve_quadratic_eqn(cell_temp, current, imp0, c0, c1, alpha_imp, t_ref=25):
    """
    Solves for x given ax**2 + bx + c = 0 where a, b, and c are arrays
    """
    discriminant = (_b_func(cell_temp,
                            imp0, c0,
                            alpha_imp))**2 - 4*_a_func(cell_temp,
                                                       imp0, c1,
                                                       alpha_imp)*_c_func(current)
    sol1 = (-_b_func(cell_temp, imp0, c0, alpha_imp, t_ref) + np.sqrt(discriminant))/(2*_a_func(cell_temp, imp0, c1, alpha_imp, t_ref))
    sol2 = (-_b_func(cell_temp, imp0, c0, alpha_imp, t_ref) - np.sqrt(discriminant))/(2*_a_func(cell_temp, imp0, c1, alpha_imp, t_ref))
    return sol1, sol2


def get_transmission(cell_temp, current, E_e, scaling_factor, imp0, c0,
                     c1, alpha_imp, t_ref=25):
    
    """
    Solves Eqn. 2 from [1] for E_e, then divides by measured plane-of-array
    irradiance (POA) to obtain a transmission coefficient T. 

    Parameters
    ==========
    cell_temp : array
        Temperature of cells inside module [degrees C]
    current : array
        [A]
    E_e : sarray
        Effective irradiance to which the PV cells in the module respond
    scaling_factor : int
        The number of strings connected in parallel to the combiner that
        measures current
    imp0 : float
        Namplate short-circuit current[A]
    c0, c1 : float
        Empirically determined coefficients relating Imp to effective
        irradiance
    alpha_imp : float
        Normalized temperature coefficient for Isc, (1/°C).
    t_ref : float
        Reference cell temperature [degrees C]

    [1] King, D.L., E.E. Boyson, and J.A. Kratochvil, Photovoltaic Array
    Performance Model, SAND2004-3535, Sandia National Laboratories,
    Albuquerque, NM, 2004.
    """
    T1, T2 = _solve_quadratic_eqn(cell_temp,
                                  current/scaling_factor,
                                  imp0,
                                  c0,
                                  c1,
                                  alpha_imp, t_ref=t_ref)/(E_e)
    
    T1[np.argwhere(current == 0)] = 0
    T2[np.argwhere(current == 0)] = 0

    T1[np.argwhere(np.isnan(current))] = np.nan
    T2[np.argwhere(np.isnan(current))] = np.nan

    T1[T1 < 0] = np.nan
    T2[T2 < 0] = np.nan

    T1[T1 > 1] = 1
    T2[T2 > 1] = 1

    return T1, T2
