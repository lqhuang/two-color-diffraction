import os
import time

import numpy as np
from matplotlib import pyplot as plt

from PeakList import PeakList
import utils



if __name__ == '__main__':
    run_folder = '/Users/lqhuang/Documents/CSRC/Data/Two-color/new_peak_lists/run78/'
    run_ID = 78
    exp = PeakList(run_ID, run_folder, isfilter=False)

    exp.plot_ewald_construction
    exp.plot_q_Intensity()
    exp.plot_pixelmap()
    exp.plot_satellite()
    exp.plot_intensity_ratio()
    exp.plot_ratio_hist()
    
    # plt.show()
