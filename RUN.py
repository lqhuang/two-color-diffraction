import os

import numpy as np

class PeakList:
    RUN_ID = 0
    HIT_NUM = 0
    RUN_FOLDER = ''
    FILTER = 0
    FILTER_SECTION = ''

    def __init__(self, ID, folder, parameters):
        """
        initial class
        """
        self.RUN_ID = folder
        fname_list = [name for name in os.listdir(run_folder) if bool(name.find('txt'))]
        
    def __str__(self):
        """Print information of Class"""
        self
 
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

    def get_q_vector(x, y, energy, WDshift):
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

        q = 2 * energy / HC * np.sin(0.5 * np.arctan(rho/WD)) # reciprocal vector

        qx = q * np.cos(phi)
        qy = q * np.sin(phi)

        return q, qx, qy