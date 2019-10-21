# !/usr/bin/env python3

# import os
# import sys
# import shutil
# import subprocess
# import re
# import datetime
# import time
import logging
import numpy as np

#  ignore numpy warnings
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


def info(y, name="", print_output=True):
    ''' print info about numpy array'''
    if isinstance(y, (np.ndarray, np.generic)):
        out = f"{str(name)}\t{y.dtype}\t{str(y.shape)}\
                \t<{y.min():.3f} {y.mean():.3f} {y.max():.3f}> \
                ({y.std():.3f})\t{type(y)} "
    else:
        out = f"{name}\t// {type(y)}"
    if print_output:
        print(out)
    return out


def normalize(y, inrange=None, outrange=(0, 1)):
    ''' Normalize numpy array --> values 0...1 '''

    imin, imax = inrange if inrange else (np.min(y), np.max(y))
    omin, omax = outrange
    logging.debug(f"normalize array, \
                  limits - in: {imin},{imax} out: {omin},{omax}")

    return np.clip(
            omin + omax * (y - imin) / (imax - imin), a_min=omin, a_max=omax)


def main():
    print(f"library: {__file__}")
    logging.basicConfig(
        level=10, format='!%(levelno)s [%(module)10s%(lineno)4d]\t%(message)s')


if __name__ == '__main__':
    main()
