# !/usr/bin/env python3

# import os
# import sys
# import shutil
# import subprocess
# import re
# import datetime
# import time
import logging

#  ignore numpy warnings
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


# JPEGTRAN_EXE = 'jpegtran-droppatch'  #  http://jpegclub.org/jpegtran/
MAGICK_EXE = 'convert'
# EXIFTOOL_EXE = 'exiftool'


#  ---------------- INIT -------------------------------
# SC_DIR, SC_NAME = os.path.split(os.path.realpath(__file__))
logging.debug(f".... modul: {__file__}")


def main():
    print(f"library: {__file__}")
    logging.basicConfig(
        level=10, format='!%(levelno)s [%(module)10s%(lineno)4d]\t%(message)s')


if __name__ == '__main__':
    main()
