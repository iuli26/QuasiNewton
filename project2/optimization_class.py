import numpy as np
# from helper_functions import gradient_approx, Hessian
from numpy.linalg import inv, norm
from scipy.linalg import cholesky

def is_pos_def(A):
    try:
        _ = cholesky(A, lower=True)
        return True
    except np.linalg.LinAlgError:
        return False

class Problem:
    def __init__(self, obj_func, derivative):
        self.f = obj_func
        self.df = derivative if derivative is not None else self.gradient_approx
        self.hess = self.Hessian
    
    def gradient_approx(self, x, h=1e-6):
        n = len(x)
        grad = np.zeros(n)
        for j in range(n):
            dx = np.zeros(n)
            dx[j] = h
            grad[j] = (self.f(x+dx) - self.f(x-dx))/(2*h)
        return grad
    
    def Hessian(self, x, h=1e-6):

        n = len(x)
        hess = np.zeros((n, n))
        for j in range (n):
            dxj = np.zeros(n)
            dxj[j] = h
            for k in range(n):
                dxk = np.zeros(n)
                dxk[k] = h
                hess[j][k] = 1/(4*h**2) * (self.f(x+dxj+dxk) - self.f(x - dxk + dxj) - self.f(x+dxk-dxj) + self.f(x-dxk-dxj))

        hess = 0.5 * (hess + hess.T)
        return hess


class Optimization_method:

    """
    General Optimization class that does the 'optimization process' aka updating the x
    """
    def __init__(self, problem, x0, alpha0, tol, line_search, max_iterations=5000):
        self.max_iterations = max_iterations
        self.tol = tol
        self.x = x0
        self.f = problem.f
        self.df = problem.df
        self.alpha = alpha0
        self.line_search = line_search
        self.prob_class = problem
        self.hess = problem.hess

        self.history = None
        self.it = 0
    

    def step(self):
        pass

    def optimization_process(self):
        self.history = [self.x.copy()]     # an array that keeps track of the solution evolution
        for k in range(self.max_iterations):
            s = self.step()
            self.x += self.alpha*s
            self.history.append(self.x.copy())

            if norm(s) < self.tol or norm(self.df(self.x)) < self.tol:
                print("Our optimization terminated.")
                break
            self.it += 1


        self.history = np.array(self.history)

        return self.x, self.f(self.x), self.history, self.it



class Newtons_method(Optimization_method):

    """
    Newton's method (derived from the Optimization class)

    """

    def __init__(self, problem, x0, alpha0, tol, line_search):
        super().__init__(problem, x0, alpha0, tol, line_search)

        self.s = None

    def phi(self, alpha):
        return self.f(self.x + alpha*self.s)

    def d_phi(self, alpha):
        return np.dot(self.df( self.x+alpha*self.s), self.s)

    def step(self):

        g = self.df(self.x)
        G = self.hess(self.x)

        self.s = -inv(G)@g
        if np.dot(g, self.s) > 0:
            tau = .1
            while np.dot(g, self.s) > 0:
                random_step = tau * np.random.uniform (-1, 1, size=self.s.shape)
                self.x += random_step
                g = self.df( self.x)
                G = self.hess(self.x)
                self.s = -inv(G) @ g

        if self.line_search == 'exact': self.alpha = self.exact_line_search(self.prob_class, self.x, self.s)
        elif self.line_search == 'inexact': self.alpha = self.inexact_line_search()
        else: self.alpha = 1

        return self.s


    def exact_line_search(self, problem, x, p, tol=1e-10, max_iter=60, max_expand=20):
        x = np.asarray(x, dtype=float)
        p = np.asarray(p, dtype=float)

        def dphi(a: float) -> float:
            return float(np.dot(self.df(x + a * p), p))

        a0, a1 = 0.0, 1.0
        f0, f1 = dphi(a0), dphi(a1)

        for _ in range(max_expand):
            if f0 * f1 <= 0.0:
                break
            a1 *= 2.0
            f1 = dphi(a1)

        if f0 * f1 > 0.0:
            return 1.0  

        lo, hi = (a0, a1) if f0 < 0.0 else (a1, a0)
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            fm = dphi(mid)
            if abs(fm) <= tol:
                return mid
            if fm > 0.0:
                hi = mid
            else:
                lo = mid
        return 0.5 * (lo + hi)


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

    """
    Quasi-Newton class derived fron Newton class
    """
    def __init__(self, problem, x0, alpha0, tol, line_search):
        super ().__init__ (problem, x0, alpha0, tol, line_search)

        self.H = np.eye(len(x0), dtype=float)
        print("Is the first Hessian positive definite?", is_pos_def(self.H))

    def inverse_hess(self):
        pass

    def step(self):

        g = self.df(self.x)


        if self.it > 0:
            self.H = self.inverse_hess()

        self.s = -1 * self.H @ g

        if np.dot(g, self.s) > 0:
            #print ("NOT descent direction")
            self.s = -g

        #     # Alternative: Take a small random step until we get a descent direction
            # tau = .1
            # while np.dot(g, self.s) > 0:
            
            #     random_step = tau * np.random.uniform (-1, 1, size=self.s.shape)
            #     self.x += random_step
            #     g = gradient (self.f, self.x)
            #     self.H = inv (Hessian (self.f, self.x))
            #     self.s = -1 * self.H @ g


        if self.line_search == 'exact': self.alpha = self.exact_line_search(self.prob_class, self.x, self.s)
        elif self.line_search == 'inexact': self.alpha = self.inexact_line_search()
        else: self.alpha = 1

        return self.s



class Good_Broyden(Quasi_Newton):
    def __init__(self, problem, x0, alpha0, tol, line_search):
        super ().__init__ (problem, x0, alpha0, tol, line_search)

    def inverse_hess(self):

        H_prev = self.H.copy()

        if len (self.history) < 2:
            print("Ai belit-o")
        else:
            x_prev, x_curr = self.history[-2], self.history[-1]
            g_prev, g_curr = self.df( x_prev), self.df( x_curr)

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
        #self.H = inv(self.hess(self.x))
    def inverse_hess(self):

        H_prev = self.H.copy()

        if len (self.history) < 2:
            print("Ai belit-o")
        else:
            x_prev, x_curr = self.history[-2], self.history[-1]
            g_prev, g_curr = self.df( x_prev), self.df( x_curr)

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
            g_prev, g_curr = self.df( x_prev), self.df( x_curr)

            gamma = g_curr - g_prev
            delta = x_curr - x_prev

            u = delta - H_prev@gamma
            a = 1/(np.dot(u, gamma))

            H_curr = H_prev + a * np.outer(u, u)

            return H_curr


class DFP(Quasi_Newton):

    def __init__(self, problem, x0, alpha0, tol, line_search):
        super ().__init__ (problem, x0, alpha0, tol, line_search)


    def inverse_hess(self):
        H_prev = self.H.copy()

        if len (self.history) < 2:
            print("Ai belit-o")
        else:
            x_prev, x_curr = self.history[-2], self.history[-1]
            g_prev, g_curr = self.df( x_prev), self.df( x_curr)

            y = g_curr - g_prev
            s = x_curr - x_prev

            s = np.asarray(s).reshape(-1,1)
            y = np.asarray(y).reshape(-1,1)

            Hy = H_prev @ y

            denom1 = s.T @ y        
            denom2 = y.T @ Hy 

            term1 = np.outer(s,s.T)
            term2 = np.outer(Hy, Hy.T)

            return H_prev + term1/denom1 - term2/denom2
  


class BFGS(Quasi_Newton):

    def __init__(self, problem, x0, alpha0, tol, line_search):
        super ().__init__ (problem, x0, alpha0, tol, line_search)

    def inverse_hess(self):

        H_prev = self.H.copy()

        if len (self.history) < 2:
            print("Ai belit-o")
        else:

            x_prev, x_curr = self.history[-2], self.history[-1]

            if self.df != None:                 # if the Problem comes with an analytic self.df take that, if not approximate with the heloing function self.df()
                g_prev, g_curr = self.df(x_prev), self.df(x_curr)
            else:
                g_prev, g_curr = self.df( x_prev), self.df( x_curr)

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
        







