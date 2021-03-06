#!/usr/bin/env python3

import os
import sys
import time
import logging
from collections import UserDict


# ---------------- INIT -------------------------------
SC_DIR, SC_NAME = os.path.split(os.path.realpath(__file__))
# logging.info(".... modul imported: ", __file__)


def debug_it(func):
    """
    Function to print all args of decorated function
     - decorate suspicious function with @debug_it
   """

    def wrapper(*func_args, **func_kwargs):
        ''' '''
#        arg_names = func.__code__.co_varnames[:func.__code__.co_argcount]
#        arg_names = func.__code__.co_varnames
#        args = func_args[:len(arg_names)]
#        args = func_args
#        defaults = func.__defaults__ or ()
#        args = args + defaults[len(defaults) -\
#        (func.__code__.co_argcount - len(args)):] # tuple of args values
#        params = dict(zip(arg_names, args))
        params = dict(zip(func.__code__.co_varnames, func_args))
        logging.debug(f"CALL: {func.__name__ } {params}")
        return func(*func_args, **func_kwargs)
    return wrapper


# doesn't work when imported, must be copied
def ensure_single_instance():
    # ensure single instance
    from tendo import singleton; me = singleton.SingleInstance()




class Script():

    def __init__(self, quiet=True, verbose=False, loglevel=20, filepath=None,
                 format='!%(levelno)s [%(module)10s%(lineno)4d]\t%(message)s'):
        import logging
        self.start_time = time.time()
        self.args = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else ""
        print(60 * "=" + f"\n {sys.argv[0]} started        \n" + 60 * "=")
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
        print(
            "\n",
            60 *
            "-",
            f"\n script finished in {runtime} s : \t {sys.argv[0]} \n ",
            60 *
            "-")





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


class Config(UserDict):
    ''' Config dictionary:
        load - load from json
        save - save to json
        reset - reset to value of default dict
        '''

    def __init__(self, fpath=None, default={}):
        ''' create config, pass defalut value or filepath to load '''
        super(Config, self).__init__(default)
        self.default = dict(default)
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
                self.update(cfg)
                # super(Config, self).__init__(cfg)
                logging.debug(f"{self.fpath} json config loaded {self}")
        except Exception as e:
            print(e)
        self.save()  # save updated config

    def save(self, fpath=None):
        ''' save config dict to json file '''
        import io
        import json
        # save config to unicode json file
        if fpath:
            self.fpath = fpath
        if len(self) == 0:
            logging.debug(f"config empty, save skipped....")
            return
        with io.open(self.fpath, 'w', encoding='utf8') as f:
            json.dump(dict(self), f, indent=4, sort_keys=True, ensure_ascii=False)
        logging.debug(f"{self.fpath} json config saved {self}")

    def reset(self):
        ''' reset config dict to default values '''
        super(Config, self).__init__(self.default.copy())


class ConfigObj(dict):
    ''' Config object:
        load - load from json
        save - save to json
        reset - reset to value of default dict
        '''

    def __init__(self, fpath=None, default={}):
        ''' create config, pass defalut value or filepath to load '''
        super(ConfigObj, self).__init__(default)
        self.default = dict(default)
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
                self.update(cfg)
                logging.debug(f"{self.fpath} json config loaded {self}")
        except Exception as e:
            print(e)
        self.save()  # save updated config

    def save(self, fpath=None):
        ''' save config dict to json file '''
        import io
        import json
        # save config to unicode json file
        if fpath:
            self.fpath = fpath
        if len(self) == 0:
            logging.debug(f"config empty, save skipped....")
            return
        with io.open(self.fpath, 'w', encoding='utf8') as f:
            json.dump(dict(self), f, indent=4, sort_keys=True, ensure_ascii=False)
        logging.debug(f"{self.fpath} json config saved {self}")

    def reset(self):
        ''' reset config dict to default values '''
        self = self.default.copy()

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)



# ---------------- MAIN -------------------------------


def main():
    print("library - import qq")
    logging.basicConfig(
        level=10, format='!%(levelno)s [%(module)10s%(lineno)4d]\t%(message)s')
    # ~ c = Config("config.json",{"test":2})
    # ~ print(type(c))
    # ~ print(c.__dict__)
    # ~ c.save()
    c = ConfigObj("config.json",{"test":2})
    print(type(c))
    # ~ print(c.__dict__)
    c.save()
    print(c.test,c.default,c.fpath)
if __name__ == '__main__':
    main()
