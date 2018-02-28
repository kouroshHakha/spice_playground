import re
import numpy as np
import os
import scipy.interpolate as interp
import scipy.optimize as sciopt

def parse_output(output_path):
    ac_fname = os.path.join(output_path, 'ac.csv')
    dc_fname = os.path.join(output_path, 'dc.csv')

    if not os.path.isfile(ac_fname) or not os.path.isfile(dc_fname):
        print("ac/dc file doesn't exist: %s" % output_path)

    ac_raw_outputs = np.genfromtxt(ac_fname, skip_header=1)
    dc_raw_outputs = np.genfromtxt(dc_fname, skip_header=1)
    print (dc_raw_outputs)
    freq = ac_raw_outputs[:, 0]
    vout = ac_raw_outputs[:, 1]
    ibias = -dc_raw_outputs[1]

    print(freq.shape)
    print(vout.shape)
    print(ibias)

if __name__ == '__main__':
    output_path = os.path.curdir
    parse_output(output_path)
