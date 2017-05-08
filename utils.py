import numpy as np

def cart2pol(x, y):
    """
    Transform Cartesian to polar coordinates
    """
    x = np.asarray(x)
    y = np.asarray(y)
    theta = np.arctan2(y, x)
    rho = np.sqrt(x**2 + y**2)
    return theta, rho

def pol2cart(theta, rho):
    """
    Transform polar to Cartesian coordinates.
    """
    theta = np.asarray(theta)
    rho = np.asarray(rho)
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y

def get_coordinates(peak_list):
    """
    read peak list return X, Y coordinates
    Column | Value
         1 | x coordinate (pixel)
         2 | y coordiante (pixel)
         3 | Intensity
         4 | SNR
         5 | Laser Energy (eV)
         6 | working distance shift (mm)
    """
    x, y = peak_list[:, 0], peak_list[:, 1]
    return x, y

def get_polar_coordinates(peak_list):
    """
    read peak list return theta, rho in polar coordinates
    """
    x, y = peak_list[:, 0], peak_list[:, 1]
    theta, rho = cart2pol(x, y)
    return theta, rho

def get_intensity(peak_list):
    """
    Read peak list return energy.
    Column | Value
         3 | Intensity
    """
    intensity = peak_list[:,2]
    return intensity

def get_energy(peak_list):
    """
    Read peak list return energy.
    Column | Value
         5 | Laser Energy (eV)
    """
    energy = peak_list[:,4]
    return energy

def get_WDshift(peak_list):
    """
    To read peak list return Energy
    Column | Value
         6 | working distance shift (mm)
    """
    WDshift = peak_list[:,5]
    return WDshift


def get_q_vector_from_peaklist(peak_list, component=True):
    """
    Get q vector in reciprocal space
    WD = working distance

    Args:
    --------
    component: bool.
        return components of q vector or only return q vector.

    Return:
    q:
    """
    HC = float(1240 * 10) # 1240 eV*nm -> ev*angstrom
    COFFSET = 5.68e9 # 0.568 m -> angstrom

    x, y = peak_list[:, 0], peak_list[:, 1]
    energy = peak_list[:,4]
    WDshift = peak_list[:,5]
    phi, rho = cart2pol(x, y)
    rho = rho * float(110e4) # pixel * angstrom/pixel
    WD = COFFSET + WDshift * 1e7 # WDshift = mm -> angstrom

    q = 2 * energy / HC * np.sin(0.5 * np.arctan(rho/WD)) # reciprocal vector
    qx = q * np.cos(phi)
    qy = q * np.sin(phi)

    if component:
        return qx, qy
    else:
        return q


def get_q_vector(x, y, energy, WDshift, component=True):
    """
    Get q vector in reciprocal space
    WD = working distance

    Args:
        component: bool.
        return components of q vector or only return q vector.

    Return:
        q:
    """
    HC = float(1240 * 10) # 1240 eV*nm -> ev*angstrom
    COFFSET = 5.68e9 # 0.568 m -> angstrom

    phi, rho = cart2pol(x, y)
    rho = rho * float(110e4) # pixel * angstrom/pixel
    WD = COFFSET + WDshift * 1e7 # WDshift = mm -> angstrom

    q = 2 * energy / HC * np.sin(0.5 * np.arctan(rho/WD)) # reciprocal vector
    qx = q * np.cos(phi)
    qy = q * np.sin(phi)

    if component:
        return qx, qy
    else:
        return q

def get_DeltaQ(DeltaE, rho, energy, WDshift):
    """
    Get Delta Q distance

    input: ( all the input value should be the information of shorter wavelength )
        DeltaE:
        rho:
        energy:
        WDshift:

    return:
        DeltaQ:
    """
    HC = float(1240 * 10) # 1240 eV*nm -> ev*angstrom
    COFFSET = float(5.68e9) # 0.568 m -> angstrom

    rho = rho * float(110e4) # pixel * angstrom/pixel
    WD = COFFSET + WDshift * 1e7 # WDshift = mm -> angstrom

    lbda = HC / energy
    Delta_lambda = np.abs(lbda - HC/(energy-DeltaE))
    q = 2 * energy / HC * np.sin(0.5 * np.arctan(rho/WD))
    theta = np.arcsin(0.5 * q * lbda)

    DeltaQ = q * WD * Delta_lambda / (np.cos(2*theta)**2 * np.cos(theta))
    DeltaQ = DeltaQ / float(110e4) # angstrom -> pixel

    return DeltaQ

def get_DeltaE(DeltaQ, rho, energy, WDshift):
    """
    Get Delta energy
    """
    HC = float(1240 * 10) # 1240 eV*nm -> ev*angstrom
    COFFSET = 5.68e9 # 0.568 m -> angstrom

    rho = rho * float(110e4) # pixel * angstrom/pixel
    WD = COFFSET + WDshift * 1e7 # WDshift = mm -> angstrom
    DeltaQ = DeltaQ * 110e4 # pixel * angstrom/pixel
    lbda = HC / energy # lambda (angstrom)

    q = 2 * energy / HC * np.sin(0.5*np.arctan(rho/WD))
    theta = np.arcsin(0.5 * q * lbda)
    Delta_lambda = DeltaQ * (np.cos(2*theta)**2 * np.cos(theta)) / q / WD

    DeltaE = energy - HC / (lbda + Delta_lambda)

    return DeltaE
