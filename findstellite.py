import os, time

import numpy as np
from matplotlib import pyplot as plt

def all_hit(run_folder):
    """
    Find all satellite point in a Run
    Return
    """
    fname_list = [name for name in os.listdir(run_folder) if bool(name.find('txt'))]
    for i, fname in enumerate(fname_list):
        peak_list = np.loadtxt(run_folder+fname)
        find_satellite(peak_list)

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

