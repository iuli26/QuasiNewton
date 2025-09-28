import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d
from example_problems import Rosenbrock_2d
#https://mathematica.stackexchange.com/questions/143338/contour-plot-of-rosenbrock-function
def plot_Rosenbrock(steps_x, steps_y):
    N = 500

    x0_grid = np.linspace(-0.5, 2, N)
    x1_grid = np.linspace(-1.5, 4, N)


    X, Y = np.meshgrid(x0_grid, x1_grid)

    rosenbrockfunction = lambda x,y: (1-x)**2+100*(y-x**2)**2
    f = rosenbrockfunction(X, Y)


    plt.figure(figsize=(10, 8))
    plt.contour(X, Y, f, np.logspace(-0.5, 3.5, 20, base=10), cmap='gray')
    #plt.title(r'$\textrm{Rosenbrock Function: } f(x,y)=(1-x)^2+100(y-x^2)^2$')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.rc('text')
    plt.rc('font', family='serif')


    plt.plot(steps_x, steps_y, marker='o', linestyle='dashed',
          linewidth=1, markersize=5, color="red")
    plt.show()

    #
    # # Plot the surface
    # fig = plt.figure(figsize=(12, 8))
    # ax = plt.axes(projection='3d')
    #
    # surf = ax.plot_surface(X, Y, f, cmap="viridis", alpha=0.9, linewidth=0, antialiased=True)
    #
    # # Add contour projection on the bottom for context
    # ax.contour(X, Y, f, levels=np.logspace(-0.5, 3.5, 20, base=10), cmap="gray", offset=0)
    #
    # # Labels
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    # ax.set_zlabel("f(x,y)")
    # ax.set_title("Rosenbrock Function Surface")
    # X_sol, Y_sol = np.meshgrid(steps_x, steps_y)
    # sol_f = rosenbrockfunction(X_sol, Y_sol)
    # fig.colorbar(surf, shrink=0.5, aspect=10)
    # # ax.plot3D(X_sol, Y_sol, sol_f, marker='o', linestyle='dashed',
    # #        linewidth=1, markersize=8, color="red")
    # plt.show()
    #
