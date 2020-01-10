#!/usr/bin/env python3

import os
import sys
import subprocess
import re
import datetime
import logging
from pathlib import Path

# ignore numpy warnings
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")



# logging.basicConfig(level=logging.DEBUG)


# ---------------- INIT -------------------------------
SC_DIR, SC_NAME = os.path.split(os.path.realpath(__file__))
# logging.info(".... modul imported: ", __file__)


# ------------ DEBUGGING --------------------------------
# def excepthook(type_, value, tb):
# ''' run post mortem pdb on Exception '''
# import traceback
# import pdb
# traceback.print_exception(type_, value, tb)
# pdb.post_mortem(tb)
# sys.excepthook = excepthook


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
    p.add_argument('-l', '--log', help='set debug level',
                   required=False, default=False, action='store_true')
    p.add_argument(
        '-q',
        '--quiet',
        help='assume yes, do not ask for confirmation',
        required=False,
        default=False,
        action='store_true')
    p.add_argument(
        '-d',
        '--indirs',
        default=[],
        help='list of input dirs',
        required=False,
        nargs='+',
        type=str)
    p.add_argument(
        '-f',
        '--infiles',
        default=[],
        help='list of input files',
        required=False,
        nargs='+',
        type=str)
    p.add_argument('-r', '--recursive', required=False,
                   default=False, action='store_true')
    p.add_argument('-m', '--match', help='filter files by substring',
                   required=False, default="", type=str)
    p.add_argument(
        '-x',
        '--extensions',
        default=[],
        help='filter by extensions',
        required=False,
        nargs='+',
        type=str)
    p.add_argument('-i', '--ignore', help='ignore files by substring',
                   required=False, default="", type=str)
    p.add_argument(
        '-o',
        '--outdir',
        default=None,
        help='output directory',
        required=False,
        nargs='?',
        type=str)  # optional, if not set, output dir = input dir
    p.add_argument('-u', '--suffix', help='suffix to out file name',
                   default='', required=False, type=str)
    p.add_argument('-t', '--trash', help='move original images to trash',
                   required=False, default=False, action='store_true')
    args, unknown = p.parse_known_args()
    ARG = vars(args)
    if ARG['log']:
        print("Not implemeted yet")
    return ARG

# ----------------  FILES -------------------------------


def fps(indir=".", recursive=True, pattern="*", extensions=None, match=None):
    Dp = Path(indir)
    logging.debug(
        f"get files from indir: {Dp.resolve()}, recursive: {recursive}")
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
        return os.path.splitext(f)[-1] == ""
    # single extension
    if isinstance(extensions, str):
        extensions = [extensions]

    # list of extensions
    for ext in extensions:
        fileext = os.path.splitext(f)[-1]
        ext = "." + ext.strip().lstrip(".")  # ensure dot
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
    value = (re.sub(r'[^\w\s-]', '', value.decode()).strip().lower())
    value = (re.sub(r'[-\s]+', '-', value))
    return value

# ---------------- RUN -------------------------------


def runwatch(cmd, shell=False, quiet=True, verbose=False):
    ''' Run command, wait for result, print stdout live '''
    if not quiet:
        print("runwait: ", str(cmd).strip(
            r"[]").replace("'", "").replace(",", ""))
    try:
        p = subprocess.Popen(cmd,
                             shell=shell,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT)
        stdout = ""
        for line in iter(p.stdout.readline, b''):
            line = line.decode().rstrip()
            stdout += line
            print(">>> ", line)
        err = p.communicate()[1]
        if err:
            err = err.decode().strip()
        rc = p.returncode
        if rc:
            logging.info(f"rc", rc)
    except Exception as e:
        logging.warn(f"subprocess failed: \n {e}")
        raise Exception("runwait error", cmd, e)
    if not quiet:
        print("runwait cmd, stdout", cmd, stdout)
    return rc, stdout, err


def runwait(cmd, shell=False, quiet=True, verbose=False):
    ''' Run command, wait for result,
    return tuple: (returncode, stdout, stderr)
    '''
    p = subprocess.Popen(
        cmd, shell=shell, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    out, err = out.decode().strip(), err.decode().strip()
    rc = p.returncode
    if not quiet:
        print(f"run cmd: {cmd}, rc:{rc}, out:{out}, err:{err}")
    return rc, out, err


def runinbg(cmd, shell=False, quiet=True, verbose=False):
    ''' Run command in background, do not wait for result '''
    if not quiet:
        print("run in background:", cmd)
    subprocess.Popen(cmd, shell=shell, stdin=None,
                     stdout=None, stderr=None, close_fds=True)


def run(cmd, shell=False, quiet=True, verbose=False, check=True):
    ''' Run command, wait for result,
    return output only, raise if error (returncode not 0) '''
    rc, out, err = runwait(cmd, shell=shell, quiet=quiet, verbose=verbose)
    if rc and check:
        raise Exception("run error", "cmd:", cmd, "rc:",
                        rc, "out:", out, "err:", err)
    return out


def _run(*args, env=None, check=False, timeout=None):
    with subprocess.Popen([a.encode('utf-8') for a in args],
                          env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE) as p:
        try:
            stdout, stderr = p.communicate(input, timeout=timeout)
        except subprocess.TimeoutExpired:
            p.kill()
            stdout, stderr = p.communicate()
            raise subprocess.TimeoutExpired(
                p.args, timeout, output=stdout, stderr=stderr,
            )
        except BaseException:
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
        pids = [int(pid) for pid in subprocess.check_output(
            ["pidof", process_name]).split()]
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


# ---------------- TIME -------------------------------
def datestamp():
    return datetime.datetime.now().strftime("%y%m%d")


def timestamp():
    return datetime.datetime.now().strftime("%y%m%d-%H%M%S")


def datestring():
    return datetime.datetime.now().strftime("%-d.%-m.%Y")

# ---------------- MISC -------------------------------


def beep(freq=440, duration=.5, volume=.1):
    try:
        subprocess.call(
            f"play -V1 --no-show-progress --null \
            --channels 1 synth {duration} sine {freq} vol {volume} ",
            shell=True)
    except BaseException:
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
        # http:// or https:// or ftp://
        r'^(https?|ftp)://'
        # domain...
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'
        # localhost...
        r'localhost|'
        # ...or ip
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'
        # optional port
        r'(?::\d+)?'
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    # print(regex.match(url) == None)
    res = False if regex.match(url) is None else True
    return res

# ---------------- LATEX -------------------------------


def df_to_latex(df, head, column_format=None,
                columns=None, table_type="tabular"):
    '''
    convert dataframe to latex table
    lines around header and bottom, header grey rowcolor
    encessary to pass head and colum_format in parameters !!
    '''
    # import csv
    # if not set, make colum format like c*4
    column_format = column_format or f'c*{len(columns) or df.shape[1]}'
    texhead = f' \\begin{{{table_type}}} {{{column_format}}}\
    \\hline \\rowcolor{{gray!20}} {head} \\\\ \\hline \n '
    textail = f'\\hline \\end{{{table_type}}}'

    tex = df.to_csv(
        sep='đ',
        line_terminator=r' \\',
        escapechar='',
        encoding='utf-8',
        columns=columns,
        index=False,
        header=False,
        # quoting = csv.QUOTE_NONE,
        quotechar='Đ',
        na_rep="~")  # columns = export_columns

    tex = tex.replace('Đ', '')   # remove quoating
    tex = tex.replace('&', '\\&')   # escape sep
    tex = tex.replace('đ', ' & ')   # insert sep

    tex_lines = tex.split('\n')
    tex_table = texhead + ' \n'.join(tex_lines) + textail
    return tex_table


def dicts_to_csv(fp, dics):
    #    logging.debug(f"save dicts to {fp}: \n  {dics}")
    #    try:
    import csv
    with open(fp, 'w+') as f:
        w = csv.DictWriter(f, fieldnames=dics[0].keys(
        ), delimiter=',', lineterminator='\n')
        w.writeheader()
        w.writerows(dics)
#    except Exception as e:
#        logging.exception(e)


def csv_to_dicts(fp, fallback=None):
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


def replace_text_in_file(infile, outfile, replace_func):
    with open(infile) as f:
        with open(outfile, "w") as of:
            for line in f:
                of.write(replace_func(line))

# ---------------- LOG -------------------------------
# def logg(text=""):
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

# def debugg():
#    logging.basicConfig(
#        level=logging.DEBUG,
#        format='\t[%(levelname)6s]\t%(message)s'
#        )


def colorize(string, color, bold=False):
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
    if color not in colors:
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


def get_infiles(files=[], dirs=[], recursive=False, extensions=[], substr=""):
    """
    return list of filepaths from list of files and dirs
    filter list by extension or substring
    sort
    """
    print("get_infiles... ", end="", flush=True)
    # from list of files
    infiles = [os.path.abspath(f.replace('"', ''))
               for f in files]  # odstranit nadbytečné uvozovky
    # from list of dirs
    for d in dirs:
        infiles += glob_files(folder=d, extensions=extensions,
                              recursive=recursive)
    # print(infiles)
    # if none than from current dir
    if not infiles:
        infiles = glob_files(folder=os.getcwd(),
                             extensions=extensions, recursive=recursive)
        # print(infiles)
    # nothing found
    if not infiles:
        raise BaseException("no files found")

    # filter by extension
    if extensions:
        infiles = [f for f in infiles if fext(
            f).lower() in extensions]  # filter out non jpgs
        # print("filter by ext",extensions,infiles)
    # filter by substring
    if substr:
        infiles = [f for f in infiles if str(
            substr).lower() in f]  # ignore case
    # sort
    infiles = sorted(infiles)
    # print('process files: ')
    # pprint.pprint(infiles)
    print("found: ", len(infiles), infiles)
    return infiles


def get_files_by_ext(indir, inexts):
    # default dir if not passed as argument
    try:
        fps = glob_files(indir, inexts)
        print("found: ", fps)
        return fps
    except Exception as e:
        print("Error loading files:\n", e)
        sys.exit(1)


def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]


def get_files_by_extension(dp, ext):
    # dirpath, extension -> filelist (ignore case, non recursive)
    return [f for f in os.listdir(dp) if
            f.lower().endswith(ext.lower())]


def glob(folder="", extensions="", recursive=False,
         ignore_case=True, quiet=True, verbose=False):
    '''
    get files from folder by extension, if no dir given, use current dir
    example: glob_files("/home", extensions="txt",
                        recursive=True,  ignore_case=True)
    '''
    fps = []
    if not folder:
        folder = os.getcwd()
    folder = os.path.abspath(folder)
    if not quiet:
        print(f" glob_files in dir: {folder}, for exts: {extensions}")
    if not os.path.isdir(folder):
        print("folder not found", folder)
        return []

    allfps = [e.path for e in scandir(folder, recursive=recursive)]
    for fp in allfps:
        if not extensions or fits_extensions(
                fp, extensions=extensions, ignore_case=ignore_case):
            fps.append(fp)

    fps = sorted(fps)
    if not quiet:
        print(" glob_files found: " + "\t".join(fps))
    return fps


def glob_files(folder="", extensions="", recursive=False,
               ignore_case=True, quiet=True, verbose=False):
    '''
    OBSOLETE, USE glob instead
    '''
    return glob(folder=folder, extensions=extensions,
                recursive=recursive, ignore_case=ignore_case, quiet=quiet)


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
    return os.path.join(os.path.split(f)[0], str(
        prefix) + os.path.split(f)[-1])


def lowerext(f):
    # convert file extension to lowercase
    return os.path.splitext(f)[0] + os.path.splitext(f)[-1].lower()


# ---------------- MAIN -------------------------------


def main():
    print("library - import qq")
    logging.basicConfig(
        level=10, format='!%(levelno)s [%(module)10s%(lineno)4d]\t%(message)s')


if __name__ == '__main__':
    main()
