"""
Module documentation
Words Go Here
"""

def f(a: int = 3, b = None, c = None, d = None) -> None:
    """
    A print func
    @param a args
    @param b args
    @param c args
    @param d args
    """
    print(a, b, c, d, sep = '&')

def average(arg1: float, arg2: float, *args) -> float:
    """To calculate average"""
    print(args)
    sum = 0
    if args.__len__():
        for value in args:
            sum += value
    return (arg1 + arg2 + sum) / (2 + args.__len__())

class Employee:
    """class documentation"""
    def f2():
        """
        A null func
        do nothing
        return null
        """
        pass

    def tracer(func, *pargs, **kargs):
        """应用函数通用性，varargs调用语法"""
        print('calling: ', func.__name__, '(', pargs, ',', kargs, ')')
        return func(*pargs, **kargs)

    def func(a, b, c, d):
        print('calling A')
        return a + b + c +d