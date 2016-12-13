import os, sys
import argparse

import h5py
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-i", type=str, help="image file in cxi format")
parser.add_argument("-o", default="radial", type=str, help="output file prefix")
args = parser.parse_args()

filename = args.i
basename, ext = os.path.splitext(os.path.basename(filename))
print(basename)
data = h5py.File(filename,'r')

nevents = data['/cheetah/event_data/hit'][:].size

for event in range(nevents):
    output = basename + 'e' + str(event) + '.txt'
    npeaks = np.nonzero(data['entry_1/result_1/peakTotalIntensity'][event][:])[0].size