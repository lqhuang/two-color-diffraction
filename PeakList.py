import os
import glob
from functools import reduce

import numpy as np
from scipy.spatial.distance import pdist, squareform
from matplotlib import pyplot as plt

import utils


class PeakList(object):

    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['mathtext.rm'] = 'serif'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    PLOT_PAR = {'family': 'sans-serif',
                'weight': 'normal',
                'size': 14}
    plt.rc('font', **PLOT_PAR)
    plt.rc('text', **{'latex.unicode': True})

    PLOT_NUM = 0

    def __init__(self, ID, peaklist_folder, isfilter=False):
        """
        initial class
        """
        self._ID = ID
        self._FOLDER = peaklist_folder

        fname_list = glob.glob(os.path.join(peaklist_folder, '*.txt'))
        self.hit_list = [np.loadtxt(name) for name in fname_list]
        vstack = lambda x, y: np.vstack((x, y))
        self.all_hit = reduce(vstack, self.hit_list)
        self._HIT_LIST_LENGTH = len(self.hit_list)
        self._PEAK_NUMBER = len(self.all_hit)
        self.pump_points, self.probe_points = self.get_satellite()

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
        pump_points = reduce(vstack, short_wavelength_list)
        probe_points = reduce(vstack, long_wavelength_list)

        return pump_points, probe_points

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
            (shorter wavelength, longer wavelength): contanis x, y, intensity, SNR, Laser Energy, working distance shift.
            
        comments:
            pump light: shorter wavelength, higher energy
            probe light: longer wavelength
        """
        x, y = utils.get_coordinates(peak_list)
        xy = np.vstack((x, y)).T
        xy_dist_squareform = squareform(pdist(xy))
        xy_dist_tril = np.tril(xy_dist_squareform)
        idx1, idx2 = np.where(
            np.logical_and(xy_dist_tril > 0, xy_dist_tril < 40)
            )

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

            pump_points = peak_list[idx1]
            probe_points = peak_list[idx2]

            _, rho = utils.get_polar_coordinates(pump_points)
            energy = utils.get_energy(pump_points)
            WDshift = utils.get_WDshift(pump_points)
            DeltaE = 80
            DeltaQ = utils.get_DeltaQ(DeltaE, rho, energy, WDshift)
            print('DeltaQ:', DeltaQ)
            upperbound = 1.2 * DeltaQ
            lowerbound = 0.8 * DeltaQ

            pair_dist = np.abs(rho1 - rho2)
            accept_idx = np.where(
                np.logical_and(pair_dist > lowerbound, pair_dist < upperbound)
                )
            pump_points = pump_points[accept_idx]
            probe_points = probe_points[accept_idx]

            return pump_points, probe_points

    # ----------------------------------------------------------------------- #
    #                         INSTANCE METHODS                                #
    # ----------------------------------------------------------------------- #

    # ------------------------------ PLOT ------------------------------------#
    def plot_pixelmap(self):
        all_x, all_y = utils.get_coordinates(self.all_hit)

        plt.figure(self.PLOT_NUM)

        plt.plot(all_x, all_y, '.k', markersize=1)
        plt.xlabel('$x$ (pixel)')
        plt.ylabel('$y$ (pixel)')
        plt.title('Pixel Map')
        plt.axis([-900, 900, -900, 900])
        plt.tight_layout()

        self.PLOT_NUM += 1

    def plot_ewald_construction(self):
        """
        ewald sphere construction ?
        """
        all_qx, all_qy = utils.get_q_vector_from_peaklist(self.all_hit)

        plt.figure(self.PLOT_NUM)

        plt.plot(all_qx, all_qy, '.k', markersize=1)
        plt.xlabel('$q_x (\AA^{-1})$')
        plt.ylabel('$q_y (\AA^{-1})$')
        plt.title('Ewal')
        plt.tight_layout()

        self.PLOT_NUM += 1

    def plot_q_Intensity(self):

        all_q = utils.get_q_vector_from_peaklist(self.all_hit, component=False)
        all_intensity = utils.get_intensity(self.all_hit)

        plt.figure(self.PLOT_NUM)

        plt.plot(all_q, np.log(all_intensity), '.')
        plt.xlabel('$q$ ($\AA^{-1}$)')
        plt.ylabel('Intensity (log scale)')
        plt.tight_layout()

        self.PLOT_NUM += 1

    def plot_satellite(self):

        short_x, short_y = utils.get_coordinates(self.pump_points)
        long_x, long_y = utils.get_coordinates(self.probe_points)

        plt.figure(self.PLOT_NUM)

        plt.plot(short_x, short_y, '.b', markersize=1)
        plt.plot(long_x, long_y, '.r', markersize=1)
        plt.xlabel('$x$ (pixel)')
        plt.ylabel('$y$ (pixel)')
        plt.title('Two-color Diffraction Pattern')
        plt.tight_layout()

        self.PLOT_NUM += 1

    def plot_intensity_ratio(self):

        short_intensity = utils.get_intensity(self.pump_points)
        long_intensity = utils.get_intensity(self.probe_points)
        # use shorter wavelength or longer wavelength ???
        satellite_q = utils.get_q_vector_from_peaklist(
            self.pump_points, component=False)
        I_ratio = short_intensity / long_intensity

        sorted_idx = np.argsort(satellite_q)
        sorted_q = satellite_q[sorted_idx]
        sorted_ratio = I_ratio[sorted_idx]

        plt.figure(self.PLOT_NUM)

        plt.plot(sorted_q, np.log(sorted_ratio), '-o')
        plt.xlabel('$q$ ($\AA^{-1}$)')
        plt.ylabel('ratio (log scale)')
        plt.title('Intensity Ratio of Two Split Spots')
        plt.tight_layout()
        
        self.PLOT_NUM += 1

    def plot_ratio_hist(self):

        short_intensity = utils.get_intensity(self.pump_points)
        long_intensity = utils.get_intensity(self.probe_points)
        I_ratio = short_intensity / long_intensity

        plt.figure(self.PLOT_NUM)

        plt.hist(np.log(I_ratio), bins=np.arange(-6.5, 7.5), edgecolor='white')
        plt.xlabel('ratio (log scale)')
        plt.ylabel('count')
        plt.title('Histogram of Intensity Ratio')
        plt.tight_layout()

        self.PLOT_NUM += 1