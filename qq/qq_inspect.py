#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging


def f_args_log(func):
    '''Decorator to print function call details - parameters names and effective values'''
    from functools import wraps
    import logging
    @wraps(func)
    def wrapper(*func_args, **func_kwargs):
        arg_names = func.__code__.co_varnames[:func.__code__.co_argcount]
        args = func_args[:len(arg_names)]
        defaults = func.__defaults__ or ()
        args = args + defaults[len(defaults) - (func.__code__.co_argcount - len(args)):]
        params = [arg_names, args]
        args = func_args[len(arg_names):]
        if args: params.append(('args', args))
        if func_kwargs: params.append(('kwargs', func_kwargs))
        result = {"__name__": func.__name__, "func_args" : params} 
        logging.debug(result)        
        return func(*func_args, **func_kwargs)
    return wrapper  



@f_args_log
def test(a, b = 4, c = 'blah-blah', *args, **kwargs):
    pass
    
@f_args_log
def test2(a):
    pass   
    
if __name__ == "__main__":
 
    logging.basicConfig(level=10, format='!%(levelno)s [%(module)10s%(lineno)4d]\t%(message)s')  
 

    test(1)
    test2(1)
    test(1, 3)
    test(1, d = 5)
    test(1, 2, 3, 4, 5, d = 6, g = 12.9)
    f=test
    print(f.__name__)
    

