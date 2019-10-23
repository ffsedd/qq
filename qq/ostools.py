#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import shutil
import logging
from shutil import copy2
from filecmp import cmp
from qq.terminaltools import print_copy_progress
from pathlib import Path


def test(a):
    copy_pro("/home/m/temp/test.jpg", "/home/m/temp/test2.jpg", force=True)


# =========== COPY WITH PROGRESS ================================


def copy_pro(src, dst, *, follow_symlinks=True,
             force=False, progress_min_size=1e7):
    """
    Copy data from src to dst.
    show progress bar in terminal for large files (> 10 MB by default)
    If follow_symlinks is not set and src is a symbolic link, a new
    symlink will be created instead of copying the file it points to.
    """
    # dst is dir? - copy into it
    if os.path.isdir(dst):
        dst = os.path.join(dst, os.path.basename(src))

    # same file error?
    if shutil._samefile(src, dst):
        raise shutil.SameFileError("{!r} and {!r} are the same file"
                                   .format(src, dst))

    # destination exists error?
    if not force and os.path.exists(dst):
        raise IOError(f"destination exists: {dst}")

    size = os.stat(src).st_size

    # symlink copy?
    if not follow_symlinks and os.path.islink(src):
        os.symlink(os.readlink(src), dst)

    else:
        # file copy
        callback = print_copy_progress if size > progress_min_size else None
        with open(src, 'rb') as fsrc:
            with open(dst, 'wb') as fdst:
                __copyfileobj(fsrc, fdst, callback=callback, total=size)

    # copy permissions
    shutil.copymode(src, dst)

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

# ==========================================





def copy3(src, dst, trash_src=False, copy_func=copy2):
    ''' copy, skip if exists, check size '''
    from filecmp import cmp

#    copyfunc = copy_with_progress if show_progress else copy2

    logging.debug(f'copy3 {src} -> {dst}')

    src, dst = Path(src), Path(dst)

    if not dst.exists():
        copy_func(src,dst)

    # sleep(.3)

#    if os.path.getsize(src) != os.path.getsize(dst):
#        raise Exception(f'Error, copy failed {src} > {dst}')
    assert cmp(src, dst, shallow=True), f'Error, copy failed {src} > {dst}'

#    else:
#        logging.debug(f'...ok')

    if trash_src:
        logging.debug(f'...send2trash {src}')
        send2trash(str(src))

def set_readonly(fpath):
    import os
    from stat import S_IREAD, S_IRGRP, S_IROTH
    os.chmod(fpath, S_IREAD | S_IRGRP | S_IROTH)


def to_trash(fp):
    import send2trash
    o = send2trash.send2trash(str(fp))
    if o:
        logging.info(o)
    else:
        logging.info(f'trashed... {fp}')


def safe_move(src, dst):
    # MOVE DIR
    src = str(src)
    dst = str(dst)
    if os.path.isdir(src) and not os.path.isfile(dst):
        logging.info(f"SAFE MOVE DIR {src}")
        for src_dir, dirs, files in os.walk(src):
            dst_dir = src_dir.replace(src, dst, 1)
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir)
            for file_ in files:
                src_file = os.path.join(src_dir, file_)
                dst_file = os.path.join(dst_dir, file_)
                if os.path.exists(dst_file):
                    to_trash(dst_file)
                    logging.info(f"\t {dst_file} trashed")
                shutil.move(src_file, dst_dir)
                logging.info(f"{src_file} moved")
        if not os.listdir(src):
            os.rmdir(src)
            logging.info(f"{src} deleted")
    # MOVE FILE
    if os.path.isfile(src):
        logging.info(f"SAFE MOVE FILE {src}")
        if os.path.isfile(dst):
            to_trash(dst)
            shutil.move(src, dst)
        elif os.path.isdir(dst):
            shutil.move(src, dst)
    return


def zipfiles(filelist, zip_fp):
    from zipfile import ZipFile, ZIP_DEFLATED
    with ZipFile(zip_fp, 'w', ZIP_DEFLATED) as z:
        for fp in filelist:
            z.write(fp)


def rcopy(src, dest, ignore=None, overwrite=False):
    ''' recursivelly copy files '''
    from pathlib import Path
    src, dest = Path(src), Path(dest)
    ignore = ignore or []

    if src.is_dir() and str(src) not in ignore:
        dest.mkdir(exist_ok=True)
        for f in src.iterdir():
            rcopy(f, dest / f.name, ignore, overwrite=overwrite)

    elif src.is_file() and str(src) not in ignore:
        if overwrite or not dest.exists():
            shutil.copyfile(src, dest)


def make_temp_dir(name="pytemp"):
    """ make temp dir in /tmp folder, return Path """
    import tempfile
    dp = Path(tempfile.gettempdir(), name)
    dp.mkdir(parents=True, exist_ok=True)
    return dp


def connect_win_network_drive(networkPath, user=None, password=None, drive_letter=None, persistent="no"):
    winCMD = f'NET USE {drive_letter} {networkPath} /User:{user} {password} /persistent:{persistent}'
    res = run(winCMD, stdout=PIPE, shell=True)
    print(res)


if __name__ == "__main__":

    logging.basicConfig(level=10,
                        format='!%(levelno)s\
                        [%(module)10s%(lineno)4d]\t%(message)s')

    test(1)
