#!/usr/bin/env python3

import os
import sys
import shutil
import subprocess
import re
import datetime
import time
import logging
import send2trash
from pathlib import Path

from PIL import Image
# ignore numpy warnings
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")



JPEGTRAN_EXE = 'jpegtran-droppatch' # http://jpegclub.org/jpegtran/
MAGICK_EXE = 'convert'
EXIFTOOL_EXE = 'exiftool'

#logging.basicConfig(level=logging.DEBUG)



# ---------------- INIT -------------------------------
SC_DIR,SC_NAME = os.path.split(os.path.realpath(__file__))
#logging.info(".... modul imported: ", __file__)


# ------------ DEBUGGING --------------------------------
# def excepthook(type_, value, tb):
    # ''' run post mortem pdb on Exception '''
    # import traceback
    # import pdb
    # traceback.print_exception(type_, value, tb)
    # pdb.post_mortem(tb)
# sys.excepthook = excepthook


def debug_it(func):
    """
    Function to print all args of decorated function
     - decorate suspicious function with @debug_it
   """

    def wrapper(*func_args, **func_kwargs):
#        arg_names = func.__code__.co_varnames[:func.__code__.co_argcount]
#        arg_names = func.__code__.co_varnames
#        args = func_args[:len(arg_names)]
#        args = func_args
#        defaults = func.__defaults__ or ()
#        args = args + defaults[len(defaults) - (func.__code__.co_argcount - len(args)):] # tuple of args values
#        params = dict(zip(arg_names, args))
        params = dict(zip(func.__code__.co_varnames, func_args))
        logging.debug(f"CALL: {func.__name__ } {params}")
        return func(*func_args, **func_kwargs)
    return wrapper


def sleep(seconds):
    time.sleep(seconds)


class Script():

    def __init__(self,   quiet=True, verbose=False, loglevel=20, filepath=None, format='!%(levelno)s [%(module)10s%(lineno)4d]\t%(message)s'):
        import logging
        self.start_time = time.time()
        self.args = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else ""
        print(60*"="+f"\n {sys.argv[0]} started        \n"+60*"=")
        if not quiet:
            print(f" \t args: \t {self.args}")

        if filepath:
            print(f"logging to {filepath}")
            logging.basicConfig(level=loglevel,
                format=format,
                filename=filepath, filemode='w')
        else:
            logging.basicConfig(level=loglevel,
                format=format)

    def end(self):
        runtime = round(time.time() - self.start_time, 2)
        print("\n", 60*"-",f"\n script finished in {runtime} s : \t {sys.argv[0]} \n ", 60*"-")


class File(type(Path())):
    import pathlib

    _flavour = pathlib._windows_flavour if os.name == 'nt' else pathlib._posix_flavour

    def __new__(cls, *args):
        return super(File, cls).__new__(cls, *args)

    def __init__(self, *args):
        super().__init__() #Path.__init__ does not take any arg (all is done in new)
        if not self.is_file():
            raise Exception(f"Error, file not found: {self}")
        self.path = str(self.resolve())

    def with_tail(self, tail):
        return Path(self.parent) / (self.stem + tail + self.suffix)

    def lower_ext(self):
        self.rename(self.with_suffix(self.suffix.lower()))


class ImageFile(File):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def size(self):
        return Image.open(self).size

    def format(self):
        return Image.open(self).format

    @debug_it
    def to_jpg(self, out_fp=None, trash=False,  quality=85):
        logging.info(f"to_jpg {self}...")
        if self.format() == "JPEG":
            logging.info(f"... already in JPG format: {self.name} ")
            return
        if not out_fp:
            out_fp = self.with_suffix(".jpg")
        else:
            out_fp = Path(out_fp)
        Image.open(self).convert('RGB').save(out_fp,  quality=quality)
        logging.info(f"...done: {self.name} ")

        if trash:
            to_trash(self)

    @debug_it
    def to_png(self, out_fp=None, trash=False):
        logging.info(f"to_png {self}...")
        if self.format() == "PNG":
            logging.info(f"...already in PNG format: {self.name} ")
            return
        if not out_fp:
            out_fp = self.with_suffix(".png")
        else:
            out_fp = Path(out_fp)
        Image.open(self).save(out_fp)
        logging.info(f"...done: {self.name} ")
        if trash:
            to_trash(self)

    def to_png16(self, out_fp=None, trash=False):
#        import numpy as np
        from numpngw import write_png
        import skimage.io
        if not out_fp:
            out_fp = self.with_suffix(".png")
        img = skimage.io.imread(self, plugin='tifffile')
        write_png(out_fp, img)

    def read_iptc_caption(self):
        cmd = f"{EXIFTOOL_EXE} -s3  -iptc:caption-abstract '{self}' "
        try:
            p = subprocess.check_output(cmd, shell=True)
            self.iptc_caption = p.decode().strip()
        except Exception as e:
            logging.warn(f"exiftool error {cmd} {e}")
            return ""
        return self.iptc_caption


    def write_xmp_description(self, description):
        cmd = f"{EXIFTOOL_EXE} '{self}' -overwrite_original -preserve -XMP-dc:Description='{description}' "
        r = run(cmd,  shell=True)
        return r


    def write_iptc_keyword(self, tag):
        # append tag
        tag = tag.replace(" ", "_")
        cmd = f"{EXIFTOOL_EXE} '{self}' -overwrite_original -preserve -iptc:keywords-={tag} -iptc:keywords+={tag} "
        r = run(cmd, shell=True)
        return r

    def rename_by_caption(self):
        try:
            cap = self.read_iptc_caption()
            if cap == "":
                return 0, "",  None
            outfp = self.with_name(f"{cap}_#_{self.name}")
            self.rename(outfp)
            return 0, str(outfp), None # rc, out, err

        except Exception as e:
            logging.critical("failed with exception: {}".format(e))
            return 1,  str(self),  e

    @debug_it
    def update_exif_thumb(self, size=128):
        # create thumbnail
        temp_fp = Path(make_temp_dir()) / self.with_name(self.stem+'-thumb'+self.suffix)
        cmd = f'{MAGICK_EXE} "{self}" -quality 70 -thumbnail {size}x{size} "{temp_fp}" '
        o = run(cmd, shell=True, verbose=False)
        if not temp_fp.is_file():
            logging.critical("thumbnail not created, exit, code: {} \n {}".format(o, cmd))
            return 1, cmd, o
        # insert thumbnail
        cmd = f'{EXIFTOOL_EXE} -overwrite_original_in_place "-thumbnailimage<={temp_fp}" "{self}" '
        o = run(cmd, shell=True)
        # remove temp file
        try:
            temp_fp.unlink()
        except Exception as e:
            logging.warning(e)
        return 0,  o, None

    @debug_it
    def sharpen(self, out_fp=None,  radius=2.0, sigma = 1.4, percent=120, threshold=3):
        logging.info(f"sharpen {self}...")
        if not out_fp:
            out_fp = self.with_name(self.stem+"_sharp"+self.suffix)
        unsharp = f"{radius:.1f}x{sigma:.1f}+{percent/100:.2f}+{threshold/100:.2f}"
#        print ("unsharp mask:", unsharp)
        cmd = f"{MAGICK_EXE} -unsharp {unsharp} {self} {out_fp}"
        r = run(cmd,  shell=True)
        out_image = ImageFile(out_fp)
        out_image.write_xmp_description("USM_"+unsharp)    # write unsharp mask settings to XMP description
        out_image.write_iptc_keyword("sharp")    # add tag
        logging.info("...done: {self.name} ")
        return r

    @debug_it
    def crop(self, geometry, out_fp):
        opt = r' -copy all -perfect -crop {0} "{1}" "{2}" '.format(geometry, self, out_fp)
    #    verbose = " -verbose"
        cmd = JPEGTRAN_EXE + opt  # + verbose
        p = subprocess.run(cmd, shell=True)
        return p

    @debug_it
    def rotate(self, angle):
        assert(angle % 90 == 0)
        logging.info(f"rotate  {self}...")
#        perfect = "-perfect"
        if self.format() == "JPEG":
            assert(angle in [90, 180, 270])
            perfect = ""
            cmd = f'{JPEGTRAN_EXE} -copy all {perfect} -rotate {angle} -outfile "{self}" "{self}" '
            run(cmd, shell=True)
        else:
            Image.open(self).rotate(-angle).save(self) # counterclockwise
        logging.info(f"...rotated: {self.name}")


    def get_exif(self):
        with Image.open(self) as img:
            img.verify()
            exif = img._getexif()
        return exif

    @debug_it
    def resize(self, width, trash=False,  quality=80):
        logging.info(f"resize {self}...")
        bak_fp = self.with_suffix(f"{self.suffix}0")
        self.rename(bak_fp)
        with Image.open(bak_fp) as img:
            if img.size[0] > width or img.size[1] > width:
                img.thumbnail((width, width))
                img.save(self,  quality=quality)
                logging.info(f"...done: {self.name} ")
                if trash:
                    to_trash(bak_fp)
            else:
                logging.info(f"...image smaller than {width} px: {self.name} ")
                bak_fp.rename(self)

        return


    def to_trash(self):
        to_trash(self)

# ---------------- GET ARGUMENTS -------------------------------
def get_sysarg():
    if len(sys.argv) == 1:
        return 0
    elif len(sys.argv) == 2:
        return sys.argv[1]
    else:
        return sys.argv

def parse_file_arguments(p=None):
    import argparse
    if not p:
        p = argparse.ArgumentParser(description='')
    p.add_argument('-l', '--log', help='set debug level', required=False, default=False, action='store_true')
    p.add_argument('-q', '--quiet', help='assume yes, do not ask for confirmation', required=False, default=False, action='store_true')
    p.add_argument('-d', '--indirs', default=[], help='list of input dirs', required=False, nargs='+', type=str)
    p.add_argument('-f', '--infiles',  default=[], help='list of input files', required=False, nargs='+', type=str)
    p.add_argument('-r', '--recursive', required=False, default=False, action='store_true')
    p.add_argument('-m', '--match', help='filter files by substring', required=False, default="", type=str)
    p.add_argument('-x', '--extensions', default=[], help='filter by extensions', required=False, nargs='+', type=str)
    p.add_argument('-i', '--ignore', help='ignore files by substring', required=False, default="", type=str)
    p.add_argument('-o', '--outdir', default=None, help='output directory', required=False, nargs='?', type=str)   # optional, if not set, output dir = input dir
    p.add_argument('-u', '--suffix', help='suffix to out file name', default = '', required=False, type=str)
    p.add_argument('-t', '--trash', help='move original images to trash', required=False, default=False, action='store_true')
    args, unknown = p.parse_known_args()
    ARG = vars(args)
    if ARG['log']:
        print("Not implemeted yet")
    return ARG

# ----------------  FILES -------------------------------
def to_trash(fp):
    o = send2trash.send2trash(str(fp))
    if o:
        logging.info(o)
    else:
        logging.info(f'trashed... {fp}')

def safe_move(src,dst):
    # MOVE DIR
    src=str(src)
    dst=str(dst)
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

def fps(indir=".", recursive=True,  pattern="*", extensions=None, match=None):
    Dp = Path(indir)
    logging.debug(f"get files from indir: {Dp.resolve()}, recursive: {recursive}")
    fps = list(Dp.rglob(pattern)) if recursive else list(Dp.glob(pattern))

    if extensions:
        fps = [f for f in fps if f.suffix and f.suffix in extensions]

    if match:
        fps = filter_items(fps, match)
    # pprint(fps)
    [logging.debug(f"{f.parent}\t{f.stem}\t{f.suffix}") for f in fps]
    logging.debug(f"Found {len(fps)} files")
    return fps


def filter_items(lst, pattern):
    import re
    filtered_lst = [i for i in lst if re.search(pattern, str(i))]
    return filtered_lst


def fits_extensions(f, extensions, ignore_case=True):
    # return True if filename extension is in list
    # no extension
    if not os.path.isfile(f):
        return False
    if not extensions:
        return os.path.splitext(f)[-1]==""
    # single extension
    if isinstance(extensions , str):
        extensions = [extensions]

    # list of extensions
    for ext in extensions:
        fileext = os.path.splitext(f)[-1]
        ext = "."+ext.strip().lstrip(".")  # ensure dot
        if fileext == ext:
            return True
        elif ignore_case and fileext.lower() == ext.lower():
            return True
    return False





def concatenate_files(fps, out_fp):
    import fileinput
    with open(out_fp, 'w') as fout, fileinput.input(fps) as fin:
        for line in fin:
            fout.write(line)

def slugify(value):
    """
    Normalizes string, converts to lowercase, removes non-alpha characters,
    and converts spaces to hyphens.
    """
    import unicodedata
    value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore')
    value = (re.sub('[^\w\s-]', '', value.decode()).strip().lower())
    value = (re.sub('[-\s]+', '-', value))
    return value

# ---------------- RUN -------------------------------


def runwatch(cmd,  shell=False,   quiet=True, verbose=False):
    ''' Run command, wait for result, print stdout live '''
    if not quiet:
        print("runwait: ", str(cmd).strip(r"[]").replace("'","").replace(",",""))
    try:
        p = subprocess.Popen(cmd,
                     shell=shell,
                     stdout=subprocess.PIPE,
                     stderr=subprocess.STDOUT)
        stdout = ""
        for line in iter(p.stdout.readline, b''):
            line = line.decode().rstrip()
            stdout += line
            print(">>> ",line)
        err = p.communicate()[1]
        if err:
            err = err.decode().strip()
        rc = p.returncode
        if rc:
            logging.info(f"rc", rc)
    except Exception as e:
        logging.warn(f"subprocess failed: \n {e}")
        raise Exception("runwait error",  cmd,  e)
    if not quiet:
        print("runwait cmd, stdout",  cmd,  stdout)
    return rc, stdout, err


def runwait(cmd,  shell=False,   quiet=True, verbose=False):
    ''' Run command, wait for result, return tuple: (returncode, stdout, stderr) '''
    p = subprocess.Popen(cmd, shell=shell, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    out, err = out.decode().strip(), err.decode().strip()
    rc = p.returncode
    if not quiet:
        print(f"run cmd: {cmd}, rc:{rc}, out:{out}, err:{err}")
    return rc,  out, err

def runinbg(cmd,  shell=False,   quiet=True, verbose=False):
    ''' Run command in background, do not wait for result '''
    if not quiet:
        print("run in background:", cmd)
    subprocess.Popen(cmd, shell=shell, stdin=None, stdout=None, stderr=None, close_fds=True)

def run(cmd,  shell=False,   quiet=True, verbose=False,   check=True):
    ''' Run command, wait for result, return output only, raise if error (returncode not 0) '''
    rc, out, err = runwait(cmd,  shell=shell,  quiet=quiet, verbose=verbose)
    if rc and check:
        raise Exception("run error", "cmd:",  cmd, "rc:",  rc, "out:",  out, "err:",  err)
    return out



def _run(*args, env=None, check=False, timeout=None):
    with subprocess.Popen([a.encode('utf-8') for a in args], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as p:
        try:
            stdout, stderr = p.communicate(input, timeout=timeout)
        except subprocess.TimeoutExpired:
            p.kill()
            stdout, stderr = p.communicate()
            raise subprocess.TimeoutExpired(
                p.args, timeout, output=stdout, stderr=stderr,
            )
        except:
            p.kill()
            p.wait()
            raise
        retcode = p.poll()
        if check and retcode:
            raise subprocess.CalledProcessError(
                retcode, p.args, output=stdout, stderr=stderr,
            )
        return subprocess.CompletedProcess(p.args, retcode, stdout, stderr)

# ---------------- PROCESS ---------------------------



def get_process_ids(process_name):
    try:
        pids = [int(pid) for pid in subprocess.check_output(["pidof",process_name]).split()]
    except BaseException as e:
        print(e)
        pids = []
    return pids


def is_process_running(process_id):
    try:
        os.kill(process_id, 0)
        return True
    except OSError:
        return False


def ensure_single_instance():
    ''' will sys.exit(-1) if other instance is running '''
    from tendo import singleton  # pip install tendo
    singleton.SingleInstance()


# ---------------- TIME -------------------------------
def datestamp():
    return datetime.datetime.now().strftime("%y%m%d")

def timestamp():
    return datetime.datetime.now().strftime("%y%m%d-%H%M%S")

def datestring():
    return datetime.datetime.now().strftime("%-d.%-m.%Y")

# ---------------- MISC -------------------------------
def make_temp_dir(name="pytemp"):
    """ make temp dir in /tmp folder, return Path """
    import tempfile
    dp = Path(tempfile.gettempdir(), name)
    dp.mkdir(parents=True, exist_ok=True)
    return dp


def beep(freq=440, duration=.5, volume=.1):
    try:
        subprocess.call(f"play -V1 --no-show-progress --null --channels 1 synth {duration} sine {freq} vol {volume} ", shell=True)
    except:
        pass
    # requires ubuntu sox package - sudo apt install sox



def error_beep(count=2):
    for i in count:
        beep(880, 0.05)

def notify(title="", message="hey", icon_fp=None, timeout=3):
    icon_cmd = f" -i {icon_fp} " if icon_fp else "" 
    cmd = f"notify-send {icon_cmd} '{title}' '{message}' -t {timeout*1000}"
    # print(cmd)
    runinbg(cmd, shell=True)

def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    ''' key for sorted function to get natural sort, eg 2.jpg, 11.jpg, ...
    use: files = sorted(files, key=natural_sort_key)
    str(s) added to support pathlib Path
    '''
    return [int(text) if text.isdigit() else text.lower()
            for text in _nsre.split(str(s))]


# ----------------  clipboard  -------------------------------

def get_clipboard():
    from tkinter import Tk
    root = Tk()
    root.withdraw()
    return root.clipboard_get()


def set_clipboard(text):
    print(text)
    subprocess.call(f"echo {text} | xclip -selection clipboard", shell=True)


def is_valid_url(url):
    import re
    regex = re.compile(
            r'^(https?|ftp)://'                                                 # http:// or https:// or ftp://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'     # domain...
            r'localhost|'                                                        # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'                                       # ...or ip
            r'(?::\d+)?'                                                         # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    # print(regex.match(url) == None)
    res = False if regex.match(url) is None else True
    return res

# ---------------- LATEX -------------------------------

def df_to_latex(df, head, column_format=None, columns=None, table_type = "tabular"):
    '''
    convert dataframe to latex table
    lines around header and bottom, header grey rowcolor
    encessary to pass head and colum_format in parameters !!
    '''
    # import csv
    column_format = column_format or f'c*{len(columns) or df.shape[1]}'  # if not set, make colum format like c*4
    texhead = f' \\begin{{{table_type}}} {{{column_format}}} \\hline \\rowcolor{{gray!20}} {head} \\\\ \\hline \n '
    textail = f'\\hline \\end{{{table_type}}}'

    tex = df.to_csv(
        sep = 'đ',
        line_terminator = r' \\',
        escapechar = '',
        encoding='utf-8',
        columns = columns,
        index = False,
        header = False,
        # quoting = csv.QUOTE_NONE,
        quotechar = 'Đ',
        na_rep = "~")  # columns = export_columns

    tex = tex.replace('Đ','')   # remove quoating
    tex = tex.replace('&','\\&')   # escape sep
    tex = tex.replace('đ',' & ')   # insert sep

    tex_lines =  tex.split('\n')
    tex_table =  texhead + ' \n'.join(tex_lines) + textail
    return tex_table







def dicts_to_csv(fp, dics):
#    logging.debug(f"save dicts to {fp}: \n  {dics}")
#    try:
    import csv
    with open(fp, 'w+') as f:
        w = csv.DictWriter(f, fieldnames=dics[0].keys(),  delimiter=',', lineterminator='\n')
        w.writeheader()
        w.writerows(dics)
#    except Exception as e:
#        logging.exception(e)


def csv_to_dicts(fp,  fallback=None):
#    logging.debug(f"load dicts from {fp}")
    try:
        import csv
        with open(fp, 'r') as f:
            r = csv.DictReader(f, delimiter=',', lineterminator='\n')
            dics = [dict(od) for od in r]
            logging.debug(f"dics {dics}")
        return dics
    except IOError:
        print('file not found')
        if fallback:        # default list of dicts to use on error
            return fallback


def json_to_dic(fp, fallback=None):
    import json
    try:
        if Path(fp).is_file():
            with open(fp, "r") as f:
                dic = json.load(f)
        else:
            dic = fallback
    except Exception as e:
        logging.warning(f"Json load from {fp} failed: \n {e}")
        dic = fallback
    return dic


def dic_to_json(dic, fp):
    import json
    with open(fp, "w+") as f:
        json.dump(dic, f)

# ---------------- STRINGS -------------------------------
def list_to_string(l):
    return str(l).strip("[]")


def replace_text_in_file(infile,outfile,replace_func):
    with open(infile) as f:
        with open(outfile,"w") as of:
            for line in f:
                of.write(replace_func(line))

# ---------------- LOG -------------------------------
#def logg(text=""):
#    """Automatically log the current function details."""
#    import inspect
#    import logging
#    # Get the previous function, not this
#    fback = inspect.currentframe().f_back
#    logging.info("{} {} {} {}".format(
#        #os.path.basename(func.co_filename),
#        fback.f_code.co_firstlineno,
#        fback.f_lineno,
#        fback.f_code.co_name,
#        text,
#        ))
#    logging.debug("locals:{}".format(
#         pprint.pformat(fback.f_locals)
#        ))

#def debugg():
#    logging.basicConfig(
#        level=logging.DEBUG,
#        format='\t[%(levelname)6s]\t%(message)s'
#        )




def colorize(string, color,  bold=False):
    RESET_SEQ = "\033[0m"
    BOLD_SEQ = "\033[1m"
    colors = {
            'black': '\033[90m',
            'red': '\033[91m',
            'green': '\033[92m',
            'yellow': '\033[93m',
            'blue': '\033[94m',
            'pink': '\033[95m',
            'cyan': '\033[96m',
            'white': '\033[97m',
            }
    if not color in colors:
        return string
    boldseq = BOLD_SEQ if bold else ""
    return boldseq + colors[color] + string + RESET_SEQ


# ---------------- SCRIPT -------------------------------

def confirm(quiet=False):
    if not quiet:
        if input("y = continue? else = abort \n") != "y":
            sys.exit(0)




def public_dir(obj):
    ''' list public object attributes (exclude atr. starting with _) '''
    return [a for a in dir(obj) if not a.startswith('_')]





def zipfiles(filelist, zip_fp):
    from zipfile import ZipFile, ZIP_DEFLATED
    with ZipFile(zip_fp, 'w', ZIP_DEFLATED) as z:
        for fp in filelist:
            z.write(fp)


def rcopy(src, dest, ignore=None, overwrite=False):
    ''' recursivelly copy files '''
    src, dest = Path(src), Path(dest)
    ignore = ignore or []

    if src.is_dir() and str(src) not in ignore:
        dest.mkdir(exist_ok=True)
        for f in src.iterdir():
            rcopy(f, dest / f.name, ignore, overwrite=overwrite)

    elif src.is_file() and str(src) not in ignore:
        if overwrite or not dest.exists():
            shutil.copyfile(src, dest)















def get_infiles(files=[], dirs=[], recursive=False, extensions=[], substr=""):
    """
    return list of filepaths from list of files and dirs
    filter list by extension or substring
    sort
    """
    print("get_infiles... ",  end="", flush=True)
    # from list of files
    infiles = [os.path.abspath(f.replace('"','')) for f in files]  # odstranit nadbytečné uvozovky
    # from list of dirs
    for d in dirs:
        infiles += glob_files(folder=d, extensions=extensions, recursive=recursive)
    # print(infiles)
    # if none than from current dir
    if not infiles:
        infiles = glob_files(folder=os.getcwd(), extensions=extensions, recursive=recursive)
        # print(infiles)
    # nothing found
    if not infiles:
        raise BaseException("no files found")

    # filter by extension
    if extensions:
        infiles = [f for f in infiles if fext(f).lower() in extensions]  # filter out non jpgs
        # print("filter by ext",extensions,infiles)
    # filter by substring
    if substr:
        infiles = [f for f in infiles if str(substr).lower() in f]  # ignore case
    # sort
    infiles=sorted(infiles)
    # print('process files: ')
    # pprint.pprint(infiles)
    print("found: ",len(infiles), infiles)
    return infiles


def get_files_by_ext(indir, inexts):
    # default dir if not passed as argument
    try:
        fps = glob_files(indir, inexts)
        print("found: ",fps)
        return fps
    except Exception as e:
        print("Error loading files:\n",e)
        sys.exit(1)

def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]


def get_files_by_extension(dp,ext):
    # dirpath, extension -> filelist (ignore case, non recursive)
    return [f for f in os.listdir(dp) if
         f.lower().endswith(ext.lower())]


def glob(folder="", extensions="", recursive=False, ignore_case=True,   quiet=True, verbose=False):
    '''
    get files from folder by extension, if no dir given, use current dir
    example: glob_files("/home", extensions="txt", recursive=True,  ignore_case=True)
    '''
    fps = []
    if not folder:
        folder = os.getcwd()
    folder = os.path.abspath(folder)
    if not quiet:
        print(f" glob_files in dir: {folder}, for exts: {extensions}")
    if not os.path.isdir(folder):
        print ("folder not found",  folder)
        return []

    allfps = [e.path for e in scandir(folder, recursive=recursive)]
    for fp in allfps:
        if not extensions or fits_extensions(fp, extensions=extensions, ignore_case=ignore_case):
            fps.append(fp)

    fps = sorted(fps)
    if not quiet:
        print(" glob_files found: " + "\t".join(fps))
    return fps

def glob_files(folder="", extensions="", recursive=False, ignore_case=True,   quiet=True, verbose=False):
    '''
    OBSOLETE, USE glob instead
    '''
    return glob(folder=folder,  extensions=extensions, recursive=recursive, ignore_case=ignore_case,  quiet=quiet)



def scandir(path, recursive=False):
    """Recursively yield DirEntry objects for given directory."""
    for entry in os.scandir(path):
        if recursive and entry.is_dir(follow_symlinks=False):
            yield from scandir(entry.path, recursive=True)
        else:
            yield entry





# ---------------- GET FILE NAMES -------------------------------
def fpath(f):
    # full file path
    return os.path.abspath(f)
def dpath(f):
    # dir path
    return os.path.split(os.path.abspath(f))[0]
def pardpath(f):
    # parent dir path
    dirp = dpath(f)
    pardirp = dpath(dirp)
    return pardirp
def dname(f):
    # dir name
    return os.path.basename(os.path.dirname(f))
def fname(f):
    # file name with ext
    return os.path.basename(f)
def fext(f):
    # extension
    return os.path.splitext(f)[-1].lstrip(".")
def fstem(f):
    # file stem (name w/o ext)
    return os.path.splitext(os.path.basename(f))[0]
def pstem(f):
    # path stem (path + name w/o ext)
    return os.path.splitext(f)[0]
def suffix(f, suffix):
    # add suffix before file extension
    return os.path.splitext(f)[0] + str(suffix) + os.path.splitext(f)[-1]
def prefix(f, prefix):
    # add suffix before file extension
    return os.path.join ( os.path.split(f)[0] ,  str(prefix) + os.path.split(f)[-1]  )
def lowerext(f):
    # convert file extension to lowercase
    return os.path.splitext(f)[0] + os.path.splitext(f)[-1].lower()









def save_config(cfg_dict, fp):
    import json
    # save config to unicode json file
    import io
    with io.open(fp, 'w', encoding='utf8') as f:
        json.dump(cfg_dict, f, indent=4, sort_keys=True, ensure_ascii=False)



def load_config(fp, default_cfg=None):
    import json
    try:
        with open(fp) as f:
            return json.load(f)
    except Exception as e:
        print(e)
        save_config(default_cfg, fp)
        return default_cfg


class Config(dict):
    ''' load config from json file or use default value '''

    def __init__(self, default={}, fpath=None):
        ''' create config, pass defalut value or filepath to load '''
        super(Config, self).__init__(default)
        self.fpath = fpath
        if fpath:
            self.load()

    def load(self, fpath=None):
        ''' load config dict from json file '''
        import json
        if fpath:
            self.fpath = fpath
        try:
            with open(self.fpath) as f:
                cfg = json.load(f)
                super(Config, self).__init__(cfg)
                logging.debug(f"{self.fpath} json config loaded {self}")
        except Exception as e:
            print(e)
            self.save()  # if load failed, create new config file

    def save(self, fpath=None):
        ''' save config dict to json file '''
        import io
        import json
        # save config to unicode json file
        if fpath:
            self.fpath = fpath

        with io.open(self.fpath, 'w', encoding='utf8') as f:
            json.dump(self, f, indent=4, sort_keys=True, ensure_ascii=False)
        logging.debug(f"{self.fpath} json config saved {self}")

# ---------------- MAIN -------------------------------
def main():
    print("library - import qq")
    logging.basicConfig(level=10, format='!%(levelno)s [%(module)10s%(lineno)4d]\t%(message)s')


if __name__ == '__main__':
    main()


