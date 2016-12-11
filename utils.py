import numpy as np

def cart2pol(x, y):
    """
    Transform Cartesian to polar coordinates
    """
    theta = np.arctan2(y, x)
    rho = np.sqrt(x**2 + y**2)
    return theta, rho

def pol2cart(theta, rho):
    """
    Transform polar to Cartesian coordinates.
    """
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y

def get_q_vector(x, y, E, WDshift):
    """
    Get q vector in reciprocal space
    WD = working distance

    Args:

    Return:
        q:
    """
    HC = 1240 * 10 # 1240 eV*nm -> ev*angstrom
    COFFSET = 5.68e9 # 0.568 m -> angstrom

    phi, rho = cart2pol(x, y)
    rho = rho * 110e4 # pixel * angstrom/pixel
    WD = COFFSET + WDshift * 1e7 # WDshift = mm -> angstrom

    q = 2 * E / HC * np.sin(0.5 * np.arctan(rho/WD)) # reciprocal vector

    qx = q * np.cos(phi)
    qy = q * np.sin(phi)

    return q, qx, qy

def getDeltaQ(DeltaE, rho, E, WDshift):
    """
    Get DeltaQ distance
    """
    HC = 1240 * 10 # 1240 eV*nm -> ev*angstrom
    COFFSET = 5.68e9 # 0.568 m -> angstrom

    rho = rho * 110e4 # pixel * angstrom/pixel
    WD = COFFSET + WDshift * 1e7 # WDshift = mm -> angstrom

    lbda = HC / E
    Delta_lambda = np.abs(lbda - HC / (E-DeltaE))
    q = 2 * E / HC * np.sin(0.5 * np.arctan(rho/WD))
    theta = np.arcsin(0.5 * q * lbda)

    DeltaQ = q * WD * Delta_lambda / (np.cos(2*theta)**2 * np.cos(theta))
    DeltaQ = DeltaQ / 110e4 # angstrom -> pixel

    return DeltaQ

def getDeltaE(DeltaQ, rho, E, WDshift):
    """
    Get DeltaE
    """
    HC = 1240 * 10 # 1240 eV*nm -> ev*angstrom
    COFFSET = 5.68e9 # 0.568 m -> angstrom

    rho = rho * 110e4 # pixel * angstrom/pixel
    WD = COFFSET + WDshift * 1e7 # WDshift = mm -> angstrom
    DeltaQ = DeltaQ * 110e4 # pixel * angstrom/pixel
    lbda = HC / E # lambda (angstrom)

    q = 2 * E / HC * np.sin(0.5*np.arctan(rho/WD))
    theta = np.arcsin(0.5 * q * lbda)
    Delta_lambda = DeltaQ * (np.cos(2*theta)**2 * np.cos(theta)) / q / WD

    DeltaE = E - HC / (lbda+Delta_lambda)

    return DeltaE
