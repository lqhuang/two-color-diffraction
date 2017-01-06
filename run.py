import os, time

import numpy as np
from matplotlib import pyplot as plt

from PeakList import PeakList
import utils

if __name__ == '__main__':
    run_folder = '/Users/lqhuang/Documents/CSRC/Data/Two-color/new_peak_lists/run49/'
    run_ID = 49
    run = PeakList(run_ID, run_folder, isfilter=False)
    
    all_x, all_y = utils.get_coordinates(run.all_hit)
    short_x, short_y = utils.get_coordinates(run.short_wavelength_points)
    long_x, long_y = utils.get_coordinates(run.long_wavelength_points)

    plt.figure(1)
    plt.plot(all_x, all_y, '.k')

    plt.figure(2)
    plt.plot(short_x, short_y, '.b')
    plt.plot(long_x, long_y, '.r')

    plt.show()