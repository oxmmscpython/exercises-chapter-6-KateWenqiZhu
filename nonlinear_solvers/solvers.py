"""A module providing numerical solvers for nonlinear equations."""

class ConvergenceError(Exception):
    """Exception raised if a solver fails to converge."""

    pass


def newton_raphson(f, df, x_0, eps=1.0e-5, max_its=20):
    """Solve a nonlinear equation using Newton-Raphson iteration.

    Solve f==0 using Newton-Raphson iteration.

    Parameters
    ----------
    f : function(x: float) -> float
        The function whose root is being found.
    df : function(x: float) -> float
        The derivative of f.
    x_0 : float
        The initial value of x in the iteration.
    eps : float
        The solver tolerance. Convergence is achieved when abs(f(x)) < eps.
    max_its : int
        The maximum number of iterations to be taken before the solver is taken
        to have failed.

    Returns
    -------
    float
        The approximate root computed using Newton iteration.
    """
    x = x_0
    k = 1
    while (abs(f(x))> eps) & (k <= max_its):
        x = x - f(x) /df(x)
        k += 1
    if (k == max_its + 1) & (abs(f(x))> eps):
        raise ConvergenceError("Max Iteration Reached")
    else:   
        return x



def bisection(f, x_0, x_1, eps=1.0e-5, max_its=20):
    """Solve a nonlinear equation using bisection.

    Solve f==0 using bisection starting with the interval [x_0, x_1]. f(x_0)
    and f(x_1) must differ in sign.

    Parameters
    ----------
    f : function(x: float) -> float
        The function whose root is being found.
    x_0 : float
        The left end of the initial bisection interval.
    x_1 : float
        The right end of the initial bisection interval.
    eps : float
        The solver tolerance. Convergence is achieved when abs(f(x)) < eps.
    max_its : int
        The maximum number of iterations to be taken before the solver is taken
        to have failed.

    Returns
    -------
    float
        The approximate root computed using bisection.
    """
    k=1
    a = x_0
    b = x_1
    c=(a+b)/2
    if f(a)*f(b) > 0: 
        raise ValueError("Input are of the same sign")
    else:
        while (abs(f(c))> eps) & (k <= max_its):
            if (f(a)*f(c)>0):
                a=c
            else:
                b=c
            k += 1
            c=(a+b)/2
        if (k == max_its + 1) & (abs(f(c))> eps):
            raise ConvergenceError("Max Iteration Reached")
        else:  
            return c


def solve(f, df, x_0, x_1, eps=1.0e-5, max_its_n=20, max_its_b=20):
    """Solve a nonlinear equation.

    solve f(x) == 0 using Newton-Raphson iteration, falling back to bisection
    if the former fails.

    Parameters
    ----------
    f : function(x: float) -> float
        The function whose root is being found.
    df : function(x: float) -> float
        The derivative of f.
    x_0 : float
        The initial value of x in the Newton-Raphson iteration, and left end of
        the initial bisection interval.
    x_1 : float
        The right end of the initial bisection interval.
    eps : float
        The solver tolerance. Convergence is achieved when abs(f(x)) < eps.
    max_its_n : int
        The maximum number of iterations to be taken before the newton-raphson
        solver is taken to have failed.
    max_its_b : int
        The maximum number of iterations to be taken before the newton-raphson
        solver is taken to have failed.

    Returns
    -------
    float
        The approximate root.
    """
    try:
        print("Attempting division by Newton Raphson")
        return newton_raphson(f, df, x_0, eps, max_its_n)
    except ConvergenceError:
        print("Attempting division by Bisection")
        return bisection(f, x_0, x_1, eps, max_its_b)
