#!/usr/bin/env python3

import os
import logging
from pathlib import Path




class File(type(Path())):
    import pathlib

    _flavour = pathlib._windows_flavour \
        if os.name == 'nt' else pathlib._posix_flavour

    def __new__(cls, *args):
        return super(File, cls).__new__(cls, *args)

    def __init__(self, *args):
        super().__init__()  # Path.__init__ takes no arg (it does new())
        if not self.is_file():
            raise Exception(f"Error, file not found: {self}")
        self.path = str(self.resolve())

    def with_tail(self, tail):
        return Path(self.parent) / (self.stem + tail + self.suffix)

    def lower_ext(self):
        self.rename(self.with_suffix(self.suffix.lower()))

