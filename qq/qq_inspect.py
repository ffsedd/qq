#!/usr/bin/env python3
from functools import wraps
import logging


def log_func_args(func):
    '''Decorator to print function call details - parameters names and effective values'''
    @wraps(func)
    def wrapper(*func_args, **func_kwargs):
        arg_names = func.func_code.co_varnames[:func.func_code.co_argcount]
        args = func_args[:len(arg_names)]
        defaults = func.func_defaults or ()
        args = args + defaults[len(defaults) - (func.func_code.co_argcount - len(args)):]
        params = zip(arg_names, args)
        args = func_args[len(arg_names):]
        if args: params.append(('args', args))
        if func_kwargs: params.append(('kwargs', func_kwargs))
        result = {"func_name": func.func_name, "func_args" : params} 
        logging.debug(result)        
        return func(*func_args, **func_kwargs)
    return wrapper  



@log_func_args
def test(a, b = 4, c = 'blah-blah', *args, **kwargs):
    pass
    
    
    
if __name__ == "__main__":
 
    logging.basicConfig(level=10, format='!%(levelno)s [%(module)10s%(lineno)4d]\t%(message)s')  
 

    test(1)
    test(1, 3)
    test(1, d = 5)
    test(1, 2, 3, 4, 5, d = 6, g = 12.9)
    f=test
    print(f.__name__)
