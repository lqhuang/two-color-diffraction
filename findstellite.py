import os, time

import numpy as np
from matplotlib import pyplot as plt

def read_hit(peak_list):
    """
    To read hit return X, Y coordinate of Hit
    Column Value
    1. x coordinate (pixel)
    2. y coordiante (pixel)
    3. Intensity
    4. SNR
    5. Laser Energy (eV)
    6. working distance shift (mm)
    """
    x, y = peak_list[:, 0], peak_list[:, 1]
    return x, y

def find_satellite(peak_list):
    """
    Input X, Y coordinates of hit patterns
    """
    X, Y = peak_list[:, 0], peak_list[:, 1]
    E = peak_list[:, 4]
    pass

def all_hit(run_folder):
    """
    Find all satellite point in a Run
    Return
    """
    fname_list = [name for name in os.listdir(run_folder) if bool(name.find('txt'))]
    for i, fname in enumerate(fname_list):
        peak_list = np.loadtxt(run_folder+fname)
        find_satellite(peak_list)
    pass

if __name__ == '__main__':
    # plt.ion()
    run_folder = 'D:/CSRC/Projects/two-color/peak_lists/run49/'
    fname_list = [name for name in os.listdir(run_folder) if bool(name.find('txt'))]
    for i, fname in enumerate(fname_list):
        peak_list = np.loadtxt(run_folder + fname)
        x, y = read_hit(peak_list)
        plt.figure(i+1)
        plt.plot(x, y, '.k')
        plt.axis('equal')
        plt.axis([-800, 800, -800, 800])
        plt.show()
        plt.close(i+1)

