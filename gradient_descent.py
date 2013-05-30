"""
Gradient descent algorithm.
"""

def scale_add(vector, alpha, vector2):
    """
    Multiplies `alpha` by every value in `vector2`, then
    adds that result with `vector`
    """
    v3 = [alpha * item for item in vector2]
    return [x + y for x, y in zip(vector, v3)]


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

    def is_converged(self, vector1, vector2, threshold=1e-8):
        """
        Returns True if the difference of `vector1` and `vector2`
        evaluated by the `diff_func.f` is less than the threshold;
        """
        return abs(self.f(vector1) - self.f(vector2)) < threshold


def gradient_descent(initial_guess, alpha, diff_func, debug=False):

    last_guess = initial_guess

    # first iteration of gradient descent
    outputs = scale_add(last_guess, -alpha, diff_func.diff_of_f(last_guess))
    counter = 0
    while not diff_func.is_converged(last_guess, outputs):
        # set first iteration of the old value to
        last_guess = outputs
        outputs = scale_add(last_guess, -alpha, diff_func.diff_of_f(last_guess))
        counter += 1
        if debug:
            print 'iteration %d, value of f(%s) = %s' % (counter, outputs, diff_func.f(outputs))

    return outputs

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
    print gradient_descent([100, 0], .1, example_function, debug=True)
