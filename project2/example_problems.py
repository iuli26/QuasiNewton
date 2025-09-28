import numpy as np

def function_1d(x):
    output = np.sqrt(x**2 + 1)
    return output[0]


def function_2d(x):   ## f:R^n -> R ,  x in R^N
    result = (x[0] - 2)**2 + x[1]**2 - 3
    return result


def Rosenbrock_2d(x):
    return 100*(x[1] - x[0]**2)**2 + (1-x[0])**2

