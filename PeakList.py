import os
from functools import reduce

import numpy as np
from scipy.spatial.distance import pdist, squareform
from matplotlib import pyplot as plt

import utils

class PeakList():

    def __init__(self, ID, peaklist_folder, isfilter=False):
        """
        initial class
        """
        self._ID = ID
        self._FOLDER = peaklist_folder

        fname_list = [peaklist_folder+name for name in os.listdir(peaklist_folder) if name.find('txt')>0]
        self.hit_list = [np.loadtxt(name) for name in fname_list]
        vstack = lambda x, y: np.vstack((x, y))
        self.all_hit = reduce(vstack, self.hit_list)
        self._HIT_LIST_LENGTH = len(self.hit_list)
        self._PEAK_NUMBER = len(self.all_hit)
        self.short_wavelength_points, self.long_wavelength_points = self.get_satellite()
    
    def __str__(self):
        """Print information of Class"""
        print('Class PeakList \n')
        print('Run ID: ', str(self._ID), '\n')
        print('PeakList folder: ', self._FOLDER, '\n')
        print('Number of points', str(self._HIT_LIST_LENGTH), '\n')

    def get_satellite(self):
        """
        get satellite points
        """
        short_wavelength_list = list()
        long_wavelength_list = list()
        for i, hit in enumerate(self.hit_list):
            short_wavelength, long_wavelength = self.find_satellite(hit)
            if short_wavelength.size == 0:
                pass
            else:
                short_wavelength_list.append(short_wavelength)
                long_wavelength_list.append(long_wavelength)
        
        vstack = lambda x, y: np.vstack((x, y))
        short_wavelength_points= reduce(vstack, short_wavelength_list)
        long_wavelength_points = reduce(vstack, long_wavelength_list)

        return short_wavelength_points, long_wavelength_points

    @staticmethod
    def find_satellite(peak_list):
        """
        find satellite point in each hit
        input:
            peak_list: array_like, contains peak information of each hit (see the following).
            Column | Value
                1 | x coordinate (pixel)
                2 | y coordiante (pixel)
                3 | Intensity
                4 | SNR
                5 | Laser Energy (eV)
                6 | working distance shift (mm)
        return:
            (shorter wavelength, longer wavelength) contanis x, y, intensity, SNR, Laser Energy, working distance shift.
        """
        x, y = utils.get_coordinates(peak_list)
        xy = np.vstack((x, y)).T
        xy_dist_squareform = squareform(pdist(xy))
        xy_dist_tril = np.tril(xy_dist_squareform)
        idx1, idx2 = np.where(np.logical_and(xy_dist_tril>0, xy_dist_tril<40))

        point1 = peak_list[idx1]
        point2 = peak_list[idx2]

        k = (point2[:, 1] - point1[:, 1]) / (point2[:, 0] - point1[:, 0])
        b = point1[:, 1] - k * point1[:, 0]
        idx = np.where(np.logical_and(b > -100, b < 100))

        point1 = point1[idx]
        point2 = point2[idx]
        idx1 = idx1[idx]
        idx2 = idx2[idx]

        if idx1.size == 0:
            return np.array([]), np.array([])
        else:
            _, rho1 = utils.get_polar_coordinates(point1)
            _, rho2 = utils.get_polar_coordinates(point2)

            for i in range(idx1.size):
                if rho1[i] - rho2[i] > 0:
                    idx1[i], idx2[i] = idx2[i], idx1[i]
            
            short_wavelength_points = peak_list[idx1]
            long_wavelength_points = peak_list[idx2]
            
            _, rho = utils.get_polar_coordinates(short_wavelength_points)
            energy = utils.get_energy(short_wavelength_points)
            WDshift = utils.get_WDshift(short_wavelength_points)
            DeltaE = 80
            DeltaQ = utils.get_DeltaQ(DeltaE, rho, energy, WDshift)
            upperbound = 1.2 * DeltaQ
            lowerbound = 0.8 * DeltaQ

            pair_dist = np.abs(rho1 - rho2)
            accept_idx = np.where(np.logical_and(pair_dist > lowerbound, pair_dist < upperbound))
            short_wavelength_points = short_wavelength_points[accept_idx]
            long_wavelength_points = long_wavelength_points[accept_idx]

            return short_wavelength_points, long_wavelength_points