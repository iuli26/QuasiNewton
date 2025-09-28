import numpy as np
from helper_functions import gradient, Hessian
from numpy.linalg import inv, norm
from scipy.linalg import cholesky

def is_pos_def(A):

    """
    checks if a matrix is positive definite
    """
    try:
        _ = cholesky(A, lower=True)
        return True
    except np.linalg.LinAlgError:
        return False

class Problem:
    def __init__(self, obj_func, derivative):
        self.f = obj_func
        self.df = derivative


class Optimization_method:
    def __init__(self, problem, x0, alpha0, tol, line_search, max_iterations=5000):
        self.max_iterations = max_iterations
        self.tol = tol
        self.x = x0
        self.f = problem.f
        self.df = problem.df
        self.alpha = alpha0
        self.line_search = line_search

        self.history = None
        self.it = 0


    def step(self):
        pass

    def optimization_process(self):
        self.history = [self.x.copy()]     # an array that keeps track of the solution evolution
        for k in range(self.max_iterations):
            #print (f"\n=== Newton iteration {k+1} ===")
            s = self.step()
            self.x += self.alpha*s

            #print(self.x)
            self.history.append(self.x.copy())

            if norm(s) < self.tol or norm(gradient(self.f, self.x)) < self.tol:
                print("Our optimization terminated.")
                break
            self.it += 1


        self.history = np.array(self.history)

        return self.x, self.f(self.x), self.history, self.it



class Newtons_method(Optimization_method):

    def __init__(self, problem, x0, alpha0, tol, line_search):
        super().__init__(problem, x0, alpha0, tol, line_search)

        self.s = None

    def phi(self, alpha):
        return self.f(self.x + alpha*self.s)

    def d_phi(self, alpha):
        return np.dot(gradient(self.f, self.x+alpha*self.s), self.s)

    def step(self):

        if self.df != None:
            g = self.df(self.x)
        else:
            g = gradient(self.f, self.x)
        G = Hessian(self.f, self.x)


        self.s = -inv(G)@g
        if np.dot(g, self.s) > 0:
            tau = .1
            while np.dot(g, self.s) > 0:
                random_step = tau * np.random.uniform (-1, 1, size=self.s.shape)
                self.x += random_step
                g = gradient(self.f, self.x)
                G = Hessian(self.f, self.x)
                self.s = -inv(G) @ g

        if self.line_search == 'exact': self.alpha = self.exact_line_search()
        elif self.line_search == 'inexact': self.alpha = self.inexact_line_search()
        else: self.alpha = 1

        return self.s


    def exact_line_search(self):
        """
            basically another newton for minimizing f(x + alpha*s) with respect to alpha, where I computed g0 and G0 form by hand (pen and paper)

        return: alpha minimizer
        """
        alpha = self.alpha
        for k in range(self.max_iterations):
            #print("     line_search iteration:", k)
            g0 = np.dot(gradient(self.f, self.x+alpha*self.s), self.s)    ### <=> gradient(self.f, self.x).T @ self.s
            G0 = np.dot(self.s, Hessian(self.f, self.x+alpha*self.s) @ self.s)

            s0 = - g0/G0  # since both are scalars
            alpha += s0
            #print("         alpha: ", alpha)
            if abs(s0) < self.tol:
                break

        return alpha


    def inexact_line_search(self):
        sigma, rho, alpha_minus = 1e-2, 0.9, 2

        ## Armijo condition not satisfied
        while self.phi(alpha_minus) > self.phi(0) + sigma*alpha_minus*self.d_phi(0):
            alpha_minus *= 0.5

        alpha_plus = alpha_minus

        ## Armijo condition satisfied
        while self.phi(alpha_plus) <= self.phi(0) + sigma*alpha_plus*self.d_phi(0):
            alpha_plus *= 2

        ## Second condition not satisfied

        while self.d_phi(alpha_minus) < rho * self.d_phi(0):

            alpha_0 = 0.5 * (alpha_plus + alpha_minus)
            if self.phi(alpha_0) <= self.phi(0) + sigma*alpha_0*self.d_phi(0):
                alpha_minus = alpha_0
            else:
                alpha_plus = alpha_0

        alpha = alpha_minus

        return alpha



class Quasi_Newton(Newtons_method):
    def __init__(self, problem, x0, alpha0, tol, line_search):
        super ().__init__ (problem, x0, alpha0, tol, line_search)

        self.H = inv(Hessian(self.f, x0))
        print("Is the first Hessian positive definite?", is_pos_def(Hessian(self.f, x0)))

    def inverse_hess(self):
        pass

    def step(self):

        if self.df != None:         ## if the Problem comes with an analytic gradient take that, if not approximate with the helping function gradient()
           g = self.df(self.x)
        else:
           g = gradient(self.f, self.x)

        if self.it > 0:
            self.H = self.inverse_hess()

        self.s = -1 * self.H @ g

        if np.dot(g, self.s) > 0:
            #print ("NOT descent direction")
            self.s = -g

            # Alternative: Take a small random step until we get a descent direction
            #tau = .1
            # while np.dot(g, self.s) > 0:
            #
            #     random_step = tau * np.random.uniform (-1, 1, size=self.s.shape)
            #     self.x += random_step
            #     g = gradient (self.f, self.x)
            #     self.H = inv (Hessian (self.f, self.x))
            #     self.s = -1 * self.H @ g


        if self.line_search == 'exact': self.alpha = self.exact_line_search()
        elif self.line_search == 'inexact': self.alpha = self.inexact_line_search()
        else: self.alpha = 1

        return self.s



class Good_Broyden(Quasi_Newton):
    def __init__(self, problem, x0, alpha0, tol, line_search):
        super ().__init__ (problem, x0, alpha0, tol, line_search)
        #self.s = None  #### need?
        self.H = inv(Hessian(self.f, x0))


    def inverse_hess(self):

        H_prev = self.H.copy()

        if len (self.history) < 2:
            print("Ai belit-o")
        else:
            x_prev, x_curr = self.history[-2], self.history[-1]
            g_prev, g_curr = gradient(self.f, x_prev), gradient(self.f, x_curr)

            gamma = g_curr - g_prev
            delta = x_curr - x_prev

            H_gamma = H_prev @ gamma
            numerator = delta - H_gamma

            denomitor = np.dot(delta, H_gamma)
            fraction = numerator / denomitor

            outer_product = np.outer(fraction, delta)
            H_curr = H_prev + outer_product@H_prev

            return H_curr

class Bad_Broyden(Quasi_Newton):
    def __init__(self, problem, x0, alpha0, tol, line_search):
        super ().__init__ (problem, x0, alpha0, tol, line_search)

    def inverse_hess(self):

        H_prev = self.H.copy()

        if len (self.history) < 2:
            print("Ai belit-o")
        else:
            x_prev, x_curr = self.history[-2], self.history[-1]
            g_prev, g_curr = gradient(self.f, x_prev), gradient(self.f, x_curr)

            gamma = g_curr - g_prev
            delta = x_curr - x_prev

            H_gamma = H_prev @ gamma
            numerator = delta - H_gamma

            denomitor = np.dot(gamma, gamma)
            fraction = numerator / denomitor

            outer_product = np.outer(fraction, gamma)
            H_curr = H_prev + outer_product

            return H_curr


class Symmetric_Broyden(Quasi_Newton):

    def __init__(self, problem, x0, alpha0, tol, line_search):
        super ().__init__ (problem, x0, alpha0, tol, line_search)

    def inverse_hess(self):
        H_prev = self.H.copy()

        if len (self.history) < 2:
            print("Ai belit-o")
        else:
            x_prev, x_curr = self.history[-2], self.history[-1]
            g_prev, g_curr = gradient(self.f, x_prev), gradient(self.f, x_curr)

            gamma = g_curr - g_prev
            delta = x_curr - x_prev

            u = delta - H_prev@gamma
            a = 1/(np.dot(u, gamma))

            H_curr = H_prev + a * np.outer(u, u)

            return H_curr


class DFP(Quasi_Newton):

    def __init__(self, problem, x0, alpha0, tol, line_search):
        super ().__init__ (problem, x0, alpha0, tol, line_search)

        #self.s = None  #### need?
        self.H = inv(Hessian(self.f, x0))


    def inverse_hess(self):

        H_prev = self.H.copy()

        if len (self.history) < 2:
            print("Ai belit-o")
        else:
            x_prev, x_curr = self.history[-2], self.history[-1]
            g_prev, g_curr = gradient(self.f, x_prev), gradient(self.f, x_curr)

            gamma = g_curr - g_prev
            delta = x_curr - x_prev

            first_term = np.outer(delta, delta)/np.dot(delta, gamma)
            denomitor = np.dot(gamma, H_prev@gamma)
            #numerator = H_prev @ np.outer (gamma, gamma) @ H_prev
            #numerator = np.outer(H_prev@gamma, gamma) @ H_prev
            numerator = H_prev@np.outer(gamma, np.dot(gamma, H_prev))
            second_term = numerator/denomitor
            H_curr = H_prev + first_term - second_term

            return H_curr


class BFGS(Quasi_Newton):

    def __init__(self, problem, x0, alpha0, tol, line_search):
        super ().__init__ (problem, x0, alpha0, tol, line_search)

        #self.s = None  #### need?
        self.H = inv(Hessian(self.f, x0))


    def inverse_hess(self):

        H_prev = self.H.copy()

        if len (self.history) < 2:
            print("Ai belit-o")
        else:

            x_prev, x_curr = self.history[-2], self.history[-1]

            if self.df != None:                 # if the Problem comes with an analytic gradient take that, if not approximate with the heloing function gradient()
                g_prev, g_curr = self.df(x_prev), self.df(x_curr)
            else:
                g_prev, g_curr = gradient(self.f, x_prev), gradient(self.f, x_curr)

            gamma = g_curr - g_prev
            delta = x_curr - x_prev

            first_term_2 = np.outer(delta, delta)/np.dot(delta, gamma)
            first_term_1 = 1 + np.dot(gamma, H_prev@gamma)/np.dot(delta, gamma)

            first_term = first_term_1 * first_term_2

            numerator = np.outer(delta, gamma)@H_prev + H_prev@np.outer(gamma, delta)
            denomitor = np.dot(delta, gamma)
            second_term = numerator/denomitor
            H_curr = H_prev + first_term - second_term

            return H_curr





