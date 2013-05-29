from utils import scale_add, is_converged

class DiffFunc(object):
    """
    This is a base class that has
    """
    def f(self, theta=[]):
        """
        Returns the value of function `f` at vector `theta`,
        which is a scalar.
        """
        pass

    def diff_of_f(self, theta=[]):
        """
        Returns the gradient of function `self.f` evaluated at vector `theta`,
        which is a vector.
        """
        pass


def gradient_descent(initial_guess, alpha, diff_func, debug=False):

    xt_old = initial_guess

    # first iteration of gradient descent
    xt = scale_add(xt_old, -alpha, diff_func.diff_of_f(xt_old))
    n = 0
    while not is_converged(xt_old, xt, diff_func):
        # set first iteration of the old value to
        xt_old = xt
        xt = scale_add(xt_old, -alpha, diff_func.diff_of_f(xt_old))
        n += 1
        if debug:
            print 'iteration %d, value of f(%s) = %s' % (n, xt, diff_func.f(xt))

    return xt

if __name__ == '__main__':
    print
    print "EXAMPLE:"
    print """
    ExampleFunction.f takes a vector of 2 values, [x, y]
    and returns the result of (y - 6)^2 + (x - 5)^2
    """

    print """
    ExampleFunction.diff_of_f differentiates the
    function ExampleFunction.f for the vector [x, y]

in this case,

    x_diff = 2 * x - 10
    y_diff = 2 * y -12
    return [x_diff, y_diff]
    """

    print """
    Then we call gradient_descent([100, 0], .55, example_function)

    where [100, 0] is our initial guess for solving 'example_function',

    and .55 is the 'alpha'(learning rate).
    """

    class ExampleFunction(DiffFunc):
        def f(self, theta=[]):
            x, y = theta
            return ((y - 6) ** 2) + ((x - 5)**2)

        def diff_of_f(self, theta=[]):
            x, y = theta
            x_diff = 2 * x - 10
            y_diff = 2 * y -12
            return [x_diff, y_diff]

    example_function = ExampleFunction()
    print gradient_descent([100, 0], .55, example_function, debug=True)
