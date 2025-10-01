import numpy as np

from chebyquad_problem import chebyquad, gradchebyquad
from example_problems import function_1d, function_2d, Rosenbrock_2d
from plot import plot_Rosenbrock

from optimization_class import Problem, Newtons_method, Good_Broyden, Bad_Broyden, Symmetric_Broyden, DFP, BFGS
from scipy.optimize import fmin_bfgs
import time
from itertools import count

if __name__ == "__main__":


    #problem = Problem(Rosenbrock_2d, derivative=None)
    problem = Problem(chebyquad, derivative=gradchebyquad)

    n = 9
    m = n+1
    j = np.arange(1, m + 1, dtype=float)
    print(j / (m + 1.0))
    initial_guess = np.linspace(0, 1, n)
    #initial_guess = np.array([0.0, -0.5], dtype=float)     ##->>> this for the Roosebrock
    print(initial_guess)

    it = count(1)

    optimizer_configs = {
        "Newton":   Newtons_method,
        "Broyden":  Good_Broyden,
        "DFP":      DFP,
        "BFGS":     BFGS,
        "BadBroyden": Bad_Broyden,
        "SymBroyden": Symmetric_Broyden
    }

    method_name = "BFGS"

    method = optimizer_configs[method_name]
    optimizer = method(
        problem=problem,
        x0=initial_guess.copy(),
        alpha0=1,
        tol=1e-6,
        line_search='inexact'
    )

    ## ----- Our optimizer ------

    t0 = time.time()
    x, f_min, history, iterations = optimizer.optimization_process()
    steps = history.T


    print(f"{method_name} ")
    print(f"    Iterations: {iterations}")
    print(f"    f_min: {f_min :.6f}")
    print(f"    minimizer: {x}")

    print("Elapsed time: ", time.time() - t0)

    print()


    ## ----- Scipy optimizer ------


    x_opt = fmin_bfgs(
        problem.f,
        initial_guess.copy(),
        fprime=problem.df,
        disp=True,           # keep SciPy's own messages quiet (set True if you want them)
        retall=False           # set True to also get a list of all iterates
    )

    print("       minimzer:", x_opt)


    print(f'Difference between our minimizer and scipy minimizer: {np.linalg.norm(np.sort(x) - np.sort(x_opt))/np.linalg.norm(x_opt)*100 :.4f}%')

    print(problem.f(np.sort(x_opt)), problem.f(np.sort(x)))
    #plot_Rosenbrock(steps[0], steps[1])
