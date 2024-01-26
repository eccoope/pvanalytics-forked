
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
    irradiance (POA) to obtain an effective transmission [2]

    Parameters
    ----------
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

    Returns
    -------
    T1, T2 : array
        Effective transmission

    [1] King, D.L., E.E. Boyson, and J.A. Kratochvil, Photovoltaic Array
    Performance Model, SAND2004-3535, Sandia National Laboratories,
    Albuquerque, NM, 2004.
    [2] E. C. Cooper, J. L. Braid and L. M. Burnham, "Identifying the
    Electrical Signature of Snow in Photovoltaic Inverter Data," 2023 IEEE
    50th Photovoltaic Specialists Conference (PVSC), San Juan, PR, USA, 2023,
    pp. 1-5, doi: 10.1109/PVSC48320.2023.10360065.
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

def categorize(vratio, transmission, voltage, turn_on_voltage=540,
               threshold_vratio=0.93, threshold_transmission=0.59):
    
    """
    Categorizes electrical behavior into a snow-related or snow-free "mode"
    as defined in [1].

    Mode 0 = system is covered with enough opaque snow that the system is
    offline or its voltage is below the inverter's MPPT turn-on voltage
    Mode 1 = system is online and covered with non-uniform snow, such that
    both operating voltage and current are decreased by the presence of snow
    Mode 2 = system is online and covered with opaque snow, such that
    operating voltage is decreased by the presence of snow, but transmission
    is consistent with snow-free conditions
    Mode 3 = system is online and covered with light-transmissive snow, such
    that transmission is decreased but voltage is consistent with all
    system substrings being online
    Mode 4 = transmisison and voltage are consistent with snow-free conditions

    Parameters
    ----------
    vratio : float
        Ratio between measured voltage and voltage modeled using
        calculated values of transmission [dimensionless]
    transmission : float
        Fraction of irradiance measured by an onsite pyranometer that the
        array is able to utilize [dimensionless]
    voltage : float
        [V]
    turn_on_voltage : float
        The lower voltage bound on the inverter's maximum power point
        tracking (MPPT) algorithm [V]
    threshold_vratio : float
        The lower bound on vratio that is found under snow-free conditions
    threshold_transmission : float
        The lower bound on transmission that is found under snow-free
        conditions

    Returns
    -------
    mode : int

    [1] E. C. Cooper, J. L. Braid and L. M. Burnham, "Identifying the
    Electrical Signature of Snow in Photovoltaic Inverter Data," 2023 IEEE
    50th Photovoltaic Specialists Conference (PVSC), San Juan, PR, USA, 2023,
    pp. 1-5, doi: 10.1109/PVSC48320.2023.10360065.
    """
    
    if np.isnan(vratio) or np.isnan(transmission):
        return np.nan
    elif voltage < turn_on_voltage:
        return 0
    elif vratio < threshold_vratio:
        if transmission < threshold_transmission:
            return 1
        elif transmission > threshold_transmission:
            return 2
    elif vratio > threshold_vratio:
        if transmission < threshold_transmission:
            return 3
        elif transmission > threshold_transmission:
            return 4
    return np.nan