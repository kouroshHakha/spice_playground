import re
import numpy as np
import os
import scipy.interpolate as interp
import scipy.optimize as sciopt

def parse_output(self, output_path):

    raw_output = None
    for file in os.listdir(output_path):
        if file.endswith(".raw"):
            raw_output = os.path.join(output_path, file)
    if raw_output is None:
        print("something wrong with raw data:%s" % (output_path))
        return

    ac_fname = raw_output + "/ac.ac"
    dc_fname = raw_output + "/dcOp.dc"
    sci_notation = "[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?"
    freq_regex_pattern = r'\"(freq)\"\s*(%s)' % sci_notation
    vout_regex_pattern = r'\"vd\" \((%s) (%s)\)' % (sci_notation, sci_notation)
    ibias_regex_pattern = r'\"VVDD:p\" \"I\" (%s)' % sci_notation
    freq_regex = re.compile(freq_regex_pattern)
    vout_regex = re.compile(vout_regex_pattern)
    ibias_regex = re.compile(ibias_regex_pattern)

    freq = list()
    vout = list()
    ibias = None

    if not os.path.isfile(ac_fname) or not os.path.isfile(dc_fname):
        print("ac/dc file doesn't exist: %s" % raw_output)

    else:
        ac_file = open(ac_fname, 'r')
        for line in ac_file:
            freq_found = freq_regex.search(line)
            vout_found = vout_regex.search(line)
            if freq_found:
                freq.append(float(freq_found.group(2)))
            elif vout_found:
                imag = float(vout_found.group(3))
                real = float(vout_found.group(1))
                vout.append(real + imag * 1j)

        ac_file.close()
        freq = np.array(freq)
        vout = np.array(vout)

        dc_file = open(dc_fname, 'r')
        for line in dc_file:
            ibias_found = ibias_regex.search(line)
            if ibias_found:
                ibias = -float(ibias_found.group(1))
        dc_file.close()

        return freq, vout, ibias