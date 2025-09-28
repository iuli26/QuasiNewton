import numpy as np

from chebyquad_problem import chebyquad, gradchebyquad
from example_problems import function_1d, function_2d, Rosenbrock_2d
from plot import plot_Rosenbrock

from optimization_class import Problem, Newtons_method, Good_Broyden, Bad_Broyden, Symmetric_Broyden, DFP, BFGS
from scipy.optimize import fmin_bfgs
import time




if __name__ == "__main__":
    #problem = Problem(Rosenbrock_2d, derivative=None)
    problem = Problem(chebyquad, derivative=gradchebyquad)

    n = 11
    initial_guess = np.linspace(0, 1, n) #np.array([0.0, -0.5], dtype=float)

    optimizer_configs = {
        "Newton":   Newtons_method,
        "Broyden":  Good_Broyden,
        "DFP":      DFP,
        "BFGS":     BFGS,
    }

    method_name = "BFGS"

    method = optimizer_configs[method_name]
    optimizer = method(
        problem=problem,
        x0=initial_guess,
        alpha0=1,
        tol=1e-6,
        line_search='inexact'
    )


    ## ----- Our optimizer ------

    t0 = time.time()
    x, f_min, history, iterations = optimizer.optimization_process()
    steps = history.T
    #plot_Rosenbrock(steps[0], steps[1])

    print(f"{method_name} ")
    print(f"    Iterations: {iterations}")
    print(f"    f_min: {f_min :.6f}")
    print(f"    minimizer: {x}")

    print("Elapsed time: ", time.time() - t0)

    print()

    ## ----- Scipy optimizer ------


    x_opt = fmin_bfgs(problem.f, initial_guess, problem.df)
    print("       minimzer:", x_opt)
    print ()
    print(f'Difference between our minimizer and scipy minimizer:{np.linalg.norm(x-x_opt)/np.linalg.norm(x_opt) * 100 :.4f}%')


