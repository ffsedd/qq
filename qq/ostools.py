#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import shutil
import logging
from filecmp import cmp
from qq.qq_inspect import f_args_log

@f_args_log
def test(a):
    
    copy_pro("/home/m/temp/test.jpg","/home/m/temp/test2.jpg", force=True)



    
    
# =========== COPY WITH PROGRESS ================================

def _progress_percentage(perc, width=None):
    # This will only work for python 3.3+ due to use of
    # os.get_terminal_size the print function etc.

    FULL_BLOCK = '█'
    # this is a gradient of incompleteness
    INCOMPLETE_BLOCK_GRAD = ['░', '▒', '▓']

    assert(isinstance(perc, float))
    assert(0. <= perc <= 100.)
    # if width unset use full terminal
    if width is None:
        width = os.get_terminal_size().columns
    # progress bar is block_widget separator perc_widget : ####### 30%
    max_perc_widget = '[100.00%]' # 100% is max
    separator = ' '
    blocks_widget_width = width - len(separator) - len(max_perc_widget)
    assert(blocks_widget_width >= 10) # not very meaningful if not
    perc_per_block = 100.0/blocks_widget_width
    # epsilon is the sensitivity of rendering a gradient block
    epsilon = 1e-6
    # number of blocks that should be represented as complete
    full_blocks = int((perc + epsilon)/perc_per_block)
    # the rest are "incomplete"
    empty_blocks = blocks_widget_width - full_blocks

    # build blocks widget
    blocks_widget = ([FULL_BLOCK] * full_blocks)
    blocks_widget.extend([INCOMPLETE_BLOCK_GRAD[0]] * empty_blocks)
    # marginal case - remainder due to how granular our blocks are
    remainder = perc - full_blocks*perc_per_block
    # epsilon needed for rounding errors (check would be != 0.)
    # based on reminder modify first empty block shading
    # depending on remainder
    if remainder > epsilon:
        grad_index = int((len(INCOMPLETE_BLOCK_GRAD) * remainder)/perc_per_block)
        blocks_widget[full_blocks] = INCOMPLETE_BLOCK_GRAD[grad_index]

    # build perc widget
    str_perc = '%.2f' % perc
    # -1 because the percentage sign is not included
    perc_widget = '[%s%%]' % str_perc.ljust(len(max_perc_widget) - 3)

    # form progressbar
    progress_bar = '%s%s%s' % (''.join(blocks_widget), separator, perc_widget)
    # return progressbar as string
    return ''.join(progress_bar)


def _copy_progress(copied, total):
    print('\r' + _progress_percentage(100*copied/total, width=30), end='')


def _copyfile(src, dst, *, follow_symlinks=True, force=False, progress_min_size=0):
    """Copy data from src to dst.

    If follow_symlinks is not set and src is a symbolic link, a new
    symlink will be created instead of copying the file it points to.

    """
    # same file error?
    if shutil._samefile(src, dst):
        raise shutil.SameFileError("{!r} and {!r} are the same file".format(src, dst))
    
    # destination exists error?    
    if not force and os.path.exists(dst):
        raise IOError(f"destination exists: {dst}")
    
        
    # for fn in [src, dst]:
        # try:
            # st = os.stat(fn)
        # except OSError:
            # File most likely does not exist
            # pass
        # else:
            # XXX What about other special files? (sockets, devices...)
            # if shutil.stat.S_ISFIFO(st.st_mode):
                # raise shutil.SpecialFileError("`%s` is a named pipe" % fn)
    size = os.stat(src).st_size
    
    # symlink copy?
    if not follow_symlinks and os.path.islink(src):
        os.symlink(os.readlink(src), dst)
    
    else:
        # file copy
        callback = _copy_progress if size > progress_min_size else None
        with open(src, 'rb') as fsrc:
            with open(dst, 'wb') as fdst:
                __copyfileobj(fsrc, fdst, callback=callback, total=size)

    # check success by size 
    if not cmp(src, dst, shallow=True):
        raise OSError(f'Error, copy failed {src}: {size} b \
         -> {dst}: {os.stat(dst).st_size} b')
                
    return dst


def __copyfileobj(fsrc, fdst, callback, total, length=16*1024):
    copied = 0
    while True:
        buf = fsrc.read(length)
        if not buf:
            break
        fdst.write(buf)
        copied += len(buf)
        if callback:
            callback(copied, total=total)


def copy_pro(src, dst, *, follow_symlinks=True, force=False, progress_min_size=1e7):
    ''' Copy file, show progress bar in terminal for large files (> 10 MB by default) '''
    
    if os.path.isdir(dst):
        dst = os.path.join(dst, os.path.basename(src))
    _copyfile(src, dst, follow_symlinks=follow_symlinks, force=force, progress_min_size=progress_min_size)
    shutil.copymode(src, dst)
    return dst
    
# ==========================================


def copy3(src, dst):
    ''' copy, skip if exists, check size '''
    #from send2trash import send2trash
    from time import sleep
    import os
    from pathlib import Path
#    from shutil import copy2

    logging.debug(f'copy3 {src} -> {dst}')

    src, dst = Path(src), Path(dst)

    if not dst.exists():
        copy_with_progress(src,dst)
    else:
        raise IOError(f"File already exists {dst}")

    sleep(.3)

    if os.path.getsize(src) != os.path.getsize(dst):
        raise IOError(f'Error, copy failed {src} -> {dst}')
    else:
        logging.debug(f'...ok, files have same size')


def set_readonly(fpath):
    import os
    from stat import S_IREAD, S_IRGRP, S_IROTH
    os.chmod(fpath, S_IREAD|S_IRGRP|S_IROTH)
    
    
if __name__ == "__main__":
 
    logging.basicConfig(level=10, format='!%(levelno)s [%(module)10s%(lineno)4d]\t%(message)s')  
    # f=test
    # print(f.__name__, f.__code__.co_argcount)
    # print(dir(f.__code__))
    test(1)
