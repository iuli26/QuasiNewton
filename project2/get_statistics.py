import numpy as np
from chebyquad_problem import chebyquad, gradchebyquad
from optimization_class import Problem, Newtons_method, Good_Broyden, Bad_Broyden, Symmetric_Broyden, DFP, BFGS
import time
import pickle
from scipy.optimize import fmin_bfgs


if __name__ == "__main__":
    #ns = [4, 8, 11]
    ns = [5, 9, 12]

    optimizer_configs = {
#        "Broyden":  Good_Broyden,
#        "DFP":      DFP,
        "BFGS":     BFGS,
#        "BadBroyden": Bad_Broyden,
#        "SymBroyden": Symmetric_Broyden
    }

    results = []

    for n in ns:
        print(f"\n===== Dimension n = {n} =====")
        initial_guess = np.linspace(0, 1, n)
        problem = Problem(chebyquad, derivative=gradchebyquad)


        t0 = time.time ()
        x_opt = fmin_bfgs (
            problem.f,
            initial_guess.copy (),
            fprime=problem.df,
            disp=False,
            gtol=1e-6,
            full_output=False  # <- only returns the minimizer
        )
        scipy_time = time.time () - t0
        scipy_val = problem.f(x_opt)

        scipy_iters_map = {      ### I manully ran this and get the number of iterations
            # 4: 6,
            # 8: 13,
            # 11: 58
            5: 6,
            9: 28,
            12: 84
        }
        scipy_iters = scipy_iters_map[n]
        results.append ({
            "n": n,
            "method": "SciPy-BFGS",
            "line_search": "scipy",
            "iterations": scipy_iters,
            "f_min": scipy_val,
            "diff_vs_scipy": 0.0,
            "time": scipy_time
        })

        for method_name, method in optimizer_configs.items():
            optimizer = method(
                problem=problem,
                x0=initial_guess.copy(),
                alpha0=1,
                tol=1e-6,
                line_search='inexact'
            )

            t0 = time.time()
            x, f_min, history, iterations = optimizer.optimization_process()
            elapsed = time.time() - t0

            diff = np.linalg.norm(np.sort(x) - np.sort(x_opt)) / np.linalg.norm(x_opt) * 100

            results.append({
                "n": n,
                "method": method_name,
                "line_search": 'inexact',
                "iterations": iterations,
                "f_min": f_min,
                "diff_vs_scipy": diff,
                "time": elapsed
            })

    # Save results into a pickle file
    with open("optimizer_result_1.pkl", "wb") as f:
        pickle.dump(results, f)

