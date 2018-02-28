import re
import numpy as np
import os
import scipy.interpolate as interp
import scipy.optimize as sciopt
from SpiceEnv import SpiceEnv

class CSAmpEnv(SpiceEnv):

    def __init__(self, num_process, design_netlist, target_specs=None):
        SpiceEnv.__init__(self, num_process, design_netlist, target_specs=target_specs)

    def get_rewards(self, output_path):
        """

        :param output_path:
        :return
        reward: a single scalar value representing the reward value
        terminate: true if we met all the specs
        """

        bw_min = self.target_specs['bw_min']
        gain_min = self.target_specs['gain_min']
        terminate = False
        # use parse output here and also the self.target_specs dictionary that user has provided
        freq, vout, Ibias = self.parse_output(output_path)
        bw = self.find_bw(vout, freq)
        gain = self.find_dc_gain(vout)

        if (bw > bw_min and gain > gain_min):
            reward = 1
            terminate = True
        else:
            reward = - (abs(bw - bw_min) / bw_min + abs(gain - gain_min) / gain_min) * Ibias

        print('bw', bw)
        print('gain', gain)
        print ('Ibias', Ibias)

        spec = dict(
            bw=bw,
            gain=gain,
            Ibias=Ibias
        )
        return reward, terminate, spec

    def parse_output(self, output_path):

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

        return freq, vout, ibias



    def find_dc_gain (self, vout):
        return np.abs(vout)[0]

    def find_bw(self, vout, freq):
        gain = np.abs(vout)
        gain_3dB = gain[0] / np.sqrt(2)
        return self._get_best_crossing(freq, gain, gain_3dB)


    def _get_best_crossing(cls, xvec, yvec, val):
        interp_fun = interp.InterpolatedUnivariateSpline(xvec, yvec)

        def fzero(x):
            return interp_fun(x) - val

        xstart, xstop = xvec[0], xvec[-1]
        try:
            return sciopt.brentq(fzero, xstart, xstop)
        except ValueError:
            # avoid no solution
            if abs(fzero(xstart)) < abs(fzero(xstop)):
                return xstart
            return xstop